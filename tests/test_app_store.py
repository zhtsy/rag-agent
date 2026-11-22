# -*- coding: utf-8 -*-
"""app_store 的测试：账号、会话归属、消息归档。

每个用例用独立的临时 SQLite 文件，互不影响——
用 monkeypatch 改 DB_PATH 而不是环境变量，是因为模块在 import 时就把路径定下来了。
"""

import pytest

import app_store


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(app_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(app_store, "_initialized", {"done": False})
    yield
    monkeypatch.setattr(app_store, "_initialized", {"done": False})


def _user(name="alice"):
    return app_store.create_user(name, "scrypt$fake$hash", f"{name}@x.com")


# ---------------- 用户 ----------------
def test_create_and_get_user():
    u = _user()
    assert u["username"] == "alice"
    assert app_store.get_user(u["id"])["username"] == "alice"
    assert app_store.get_user_by_name("alice")["id"] == u["id"]


def test_duplicate_username_raises():
    _user()
    with pytest.raises(ValueError):
        _user()


def test_missing_user_is_none():
    assert app_store.get_user("nope") is None
    assert app_store.get_user_by_name("nobody") is None


# ---------------- 会话 ----------------
def test_create_and_list_conversations():
    u = _user()
    a = app_store.create_conversation(u["id"], "甲")
    b = app_store.create_conversation(u["id"], "乙")
    ids = [c["id"] for c in app_store.list_conversations(u["id"])]
    assert set(ids) == {a["id"], b["id"]}
    assert ids[0] == b["id"]          # 按 updated_at 倒序


def test_conversations_are_isolated_between_users():
    u1, u2 = _user("alice"), _user("bob")
    conv = app_store.create_conversation(u1["id"], "私有")
    assert app_store.list_conversations(u2["id"]) == []
    # 别人的会话读不到——这是越权的核心断言
    assert app_store.own_conversation(conv["id"], u2["id"]) is None
    assert app_store.own_conversation(conv["id"], u1["id"]) is not None


def test_rename_and_delete_only_by_owner():
    u1, u2 = _user("alice"), _user("bob")
    conv = app_store.create_conversation(u1["id"], "原标题")

    assert app_store.rename_conversation(conv["id"], u2["id"], "被别人改") is False
    assert app_store.rename_conversation(conv["id"], u1["id"], "新标题") is True
    assert app_store.get_conversation(conv["id"])["title"] == "新标题"

    assert app_store.delete_conversation(conv["id"], u2["id"]) is False
    assert app_store.get_conversation(conv["id"]) is not None
    assert app_store.delete_conversation(conv["id"], u1["id"]) is True
    assert app_store.get_conversation(conv["id"]) is None


# ---------------- 消息 ----------------
def test_append_and_read_messages_in_order():
    u = _user()
    conv = app_store.create_conversation(u["id"])
    app_store.append_message(conv["id"], "user", "问题一")
    app_store.append_message(conv["id"], "assistant", "答案一",
                             sources=["印花税法.pdf"], confidence=0.98,
                             status="answered")
    msgs = app_store.list_messages(conv["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["sources"] == []
    assert msgs[1]["sources"] == ["印花税法.pdf"]
    assert msgs[1]["confidence"] == 0.98
    assert msgs[1]["status"] == "answered"


def test_message_count_tracks_appends():
    u = _user()
    conv = app_store.create_conversation(u["id"])
    for i in range(3):
        app_store.append_message(conv["id"], "user", f"q{i}")
    listed = {c["id"]: c for c in app_store.list_conversations(u["id"])}
    assert listed[conv["id"]]["msg_count"] == 3


def test_delete_conversation_removes_messages():
    u = _user()
    conv = app_store.create_conversation(u["id"])
    app_store.append_message(conv["id"], "user", "x")
    assert app_store.delete_conversation(conv["id"], u["id"]) is True
    assert app_store.list_messages(conv["id"]) == []


def test_title_filled_from_first_message_only():
    u = _user()
    conv = app_store.create_conversation(u["id"])
    app_store.touch_conversation(conv["id"], title_if_new="首句标题")
    assert app_store.get_conversation(conv["id"])["title"] == "首句标题"
    # 第二次不应该覆盖用户后来改的标题
    app_store.rename_conversation(conv["id"], u["id"], "自定义")
    app_store.touch_conversation(conv["id"], title_if_new="另一个首句")
    assert app_store.get_conversation(conv["id"])["title"] == "自定义"


def test_stats_counts():
    u = _user()
    conv = app_store.create_conversation(u["id"])
    app_store.append_message(conv["id"], "user", "x")
    s = app_store.stats()
    assert s == {"users": 1, "conversations": 1, "messages": 1, "events": 0}
