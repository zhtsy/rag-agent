# -*- coding: utf-8 -*-
"""鉴权事件：游标语义 + UA 归类 + 真实 IP 提取。

这三块共同的特点是**错了也不报错**——返回 0、返回「未知」、返回网关 IP。
线上表现只是「没收到通知」，从日志里看不出任何异常。所以必须用断言钉住。
"""

import types

import pytest

import app_store
import auth


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(app_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(app_store, "_initialized", {"done": False})
    yield
    monkeypatch.setattr(app_store, "_initialized", {"done": False})


# ---------------------------------------------------------------- 存储层
def test_event_id_is_monotonic():
    """id 必须递增：它是轮询的唯一游标，相等或倒退都会漏事件。"""
    a = app_store.log_event("register", "alice", "1.1.1.1", "iPhone", "ua")
    b = app_store.log_event("login", "alice", "1.1.1.1", "iPhone", "ua")
    assert a > 0
    assert b > a


def test_since_cursor_is_exclusive():
    """since 是「已处理到哪」，必须严格大于，否则最后一条会反复弹。"""
    a = app_store.log_event("register", "alice")
    b = app_store.log_event("login", "bob")
    assert [e["id"] for e in app_store.list_events(since_id=a)] == [b]
    assert app_store.list_events(since_id=b) == []


def test_latest_id_is_max():
    app_store.log_event("register", "alice")
    last = app_store.log_event("login", "alice")
    assert app_store.latest_event_id() == last


def test_stats_exposes_event_count():
    """放进 /health 是为了「有没有人用过」能一眼看出来，不用另开接口翻。"""
    app_store.log_event("register", "alice")
    assert app_store.stats()["events"] == 1


def test_log_event_swallows_db_failure(monkeypatch):
    """落库失败必须返回 0 而不是抛出——记日志挂了不能让登录跟着挂。"""
    monkeypatch.setattr(app_store, "DB_PATH", "/proc/definitely-not-writable/x.db")
    monkeypatch.setattr(app_store, "_initialized", {"done": False})
    assert app_store.log_event("login", "x") == 0
    assert app_store.list_events(0) == []


# ---------------------------------------------------------------- UA 归类
@pytest.mark.parametrize("ua,want", [
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
     "iPhone"),
    # 微信内置浏览器的 UA 里同时含 iPhone/Android，先判具体才归得对
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) MicroMessenger/8.0.44",
     "微信"),
    ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36", "Android"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Windows"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15", "Mac"),
    ("curl/8.4.0", "脚本"),
    ("", "未知"),
])
def test_device_of(ua, want):
    assert auth.device_of(ua) == want


# ---------------------------------------------------------------- 真实 IP
def _req(headers=None, host="172.18.0.1"):
    return types.SimpleNamespace(
        headers={k.lower(): v for k, v in (headers or {}).items()},
        client=types.SimpleNamespace(host=host),
    )


def test_client_ip_takes_first_hop_of_xff():
    """X-Forwarded-For 是多级代理链，只有最左一段是原始客户端。"""
    assert auth.client_ip(_req({"X-Forwarded-For": "203.0.113.9, 10.0.0.1"})) == "203.0.113.9"


def test_client_ip_supports_x_real_ip():
    assert auth.client_ip(_req({"X-Real-IP": "203.0.113.7"})) == "203.0.113.7"


def test_client_ip_falls_back_to_socket_address():
    """没有代理头时回落到连接地址——容器里通常是 docker 网关 172.18.0.1。"""
    assert auth.client_ip(_req()) == "172.18.0.1"


def test_client_ip_never_raises_on_garbage():
    assert auth.client_ip(types.SimpleNamespace()) == ""
