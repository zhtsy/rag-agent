# -*- coding: utf-8 -*-
"""auth 的纯逻辑测试：哈希、令牌、输入校验、限流。

这些函数不 import 任何模型/数据库，跑一遍不到 1 秒——
把校验逻辑从接口里抽出来，就是为了能在 CI 里覆盖住。
"""

import time

import pytest

import auth


# ---------------- 密码 ----------------
def test_hash_is_not_plaintext_and_verifies():
    h = auth.hash_password("s3cret-pass")
    assert "s3cret-pass" not in h
    assert h.startswith("scrypt$")
    assert auth.verify_password("s3cret-pass", h)


def test_same_password_gets_different_hash():
    """每人一份 salt：同样的密码两次哈希不能相等，否则一眼看出谁和谁密码相同。"""
    assert auth.hash_password("same-pass") != auth.hash_password("same-pass")


def test_wrong_password_rejected():
    h = auth.hash_password("right-pass")
    assert not auth.verify_password("wrong-pass", h)
    assert not auth.verify_password("", h)


@pytest.mark.parametrize("bad", ["", "plain", "bcrypt$1$2$3$aa$bb", "scrypt$x$y$z$aa$bb"])
def test_corrupt_hash_returns_false_instead_of_raising(bad):
    """库里的哈希坏了只能"登录失败"，不能让登录接口 500。"""
    assert auth.verify_password("whatever", bad) is False


# ---------------- 输入校验 ----------------
@pytest.mark.parametrize("name", ["ab", "", "带 空格", "x" * 25, "emoji😀"])
def test_reject_bad_username(name):
    assert auth.validate_credentials(name, "longenough") is not None


@pytest.mark.parametrize("name", ["abc", "张三丰", "user_01", "a-b-c", "x" * 24])
def test_accept_username(name):
    assert auth.validate_credentials(name, "longenough") is None


def test_password_length_limits():
    assert auth.validate_credentials("goodname", "1234567") is not None
    assert auth.validate_credentials("goodname", "12345678") is None
    assert auth.validate_credentials("goodname", "x" * 129) is not None


# ---------------- 令牌 ----------------
def test_token_roundtrip():
    tok, exp = auth.issue_token("u123", ttl=60)
    assert auth.parse_token(tok) == "u123"
    assert exp > time.time()


def test_token_tampered():
    tok, _ = auth.issue_token("u123", ttl=60)
    head, tail = tok.rsplit(".", 1)
    assert auth.parse_token(head + "." + ("A" * len(tail))) is None
    # 改 uid 但沿用原签名也必须失败
    parts = tok.split(".")
    parts[1] = "attacker"
    assert auth.parse_token(".".join(parts)) is None


def test_token_expired():
    tok, _ = auth.issue_token("u123", ttl=-5)
    assert auth.parse_token(tok) is None


@pytest.mark.parametrize("bad", ["", "abc", "v2.u.1.sig", "v1..1.sig", "v1.u.notanumber.sig"])
def test_token_bad_format(bad):
    assert auth.parse_token(bad) is None


def test_signature_uses_secret(monkeypatch):
    tok, _ = auth.issue_token("u123", ttl=60)
    monkeypatch.setenv("RAG_AUTH_SECRET", "a-different-secret")
    assert auth.parse_token(tok) is None


# ---------------- 登录限流 ----------------
def test_blocks_after_five_failures_then_clears():
    key = "test|blocker"
    auth.clear_failures(key)
    for _ in range(4):
        auth.note_failure(key)
    assert auth.login_blocked(key) == 0        # 四次还没锁
    auth.note_failure(key)                     # 第五次
    assert auth.login_blocked(key) > 0
    auth.clear_failures(key)
    assert auth.login_blocked(key) == 0


def test_window_resets_counter():
    key = "test|window"
    auth.clear_failures(key)
    for _ in range(5):
        auth.note_failure(key)
    # 把首失败时间往前推，模拟"窗口已过"
    auth._FAILS[key][1] = time.time() - auth._FAIL_WINDOW - 1
    auth._FAILS[key][2] = 0
    assert auth.login_blocked(key) == 0


# ---------------- 公开注册开关 ----------------
def test_register_closed_by_default(monkeypatch):
    """默认关闭：公网部署忘了配置，不能变成开放注册。

    这是"默认值决定安全性"的典型场景——8000 端口对外，
    默认开着等于把注册权交给陌生人。
    """
    monkeypatch.delenv("RAG_ALLOW_REGISTER", raising=False)
    assert auth.allow_public_register() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_register_can_be_enabled(monkeypatch, value):
    monkeypatch.setenv("RAG_ALLOW_REGISTER", value)
    assert auth.allow_public_register() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", " Off "])
def test_register_switch_off_values(monkeypatch, value):
    """大小写与空白都要认——.env 里写成 False 也必须当关闭处理。"""
    monkeypatch.setenv("RAG_ALLOW_REGISTER", value)
    assert auth.allow_public_register() is False
