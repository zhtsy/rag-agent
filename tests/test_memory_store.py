# -*- coding: utf-8 -*-
"""memory_store 持久化层测试（需要 MySQL；连不上自动 skip）。

为什么这些测试值得写：
    "记忆"这种东西坏掉时不会报错，只会**悄悄记错**——
      · 轮次号重复 → 回填时顺序错乱，模型看到的是倒着的历史
      · 用户事实不去重 → 同一件事记十行，"喜欢川菜"和"喜欢粤菜"一起注入
      · 评测会话没拦住 → 第二次跑分带着第一次的历史，分数虚高（最难发现的假通过）
    这三类问题都只能靠测试守。
"""

import time
import uuid

import pytest

import memory_store

# 每个测试用独立前缀，互不干扰，也方便清理
_PREFIX = f"pytest-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def store():
    """连不上库就跳过——CI 里没有 MySQL 不应该红，但本地一定跑。"""
    factory = None
    for _ in range(20):          # WSL 冷启动后 MySQL 要十几秒才就绪
        factory = memory_store._get_factory()
        if factory is not None:
            break
        memory_store._last_fail_at = 0.0   # 清掉退避，允许立刻重试
        time.sleep(1)
    if factory is None:
        pytest.skip("MySQL 不可用，跳过记忆持久化测试")
    yield memory_store
    _cleanup()


def _cleanup():
    with memory_store._conn() as s:
        if s is None:
            return
        from sqlalchemy import text

        s.execute(text("DELETE FROM chat_turns WHERE session_id LIKE :p"), {"p": f"{_PREFIX}%"})
        s.execute(text("DELETE FROM chat_sessions WHERE session_id LIKE :p"), {"p": f"{_PREFIX}%"})
        s.execute(text("DELETE FROM user_memories WHERE user_id LIKE :p"), {"p": f"{_PREFIX}%"})


# =====================================================================
# 1. 开关与评测隔离
# =====================================================================
def test_eval_sessions_disabled_by_default():
    assert memory_store.enabled_for("eval-T001") is False
    assert memory_store.enabled_for("ragas-001") is False
    assert memory_store.enabled_for(f"{_PREFIX}-normal") is True


def test_eval_session_write_is_noop(store):
    sid = f"{_PREFIX}-eval"
    assert store.save_turn("eval-T001", "q", "a") == 0
    assert store.count_turns("eval-T001") == 0
    assert store.load_recent_turns("eval-T001") == []


# =====================================================================
# 2. 轮次
# =====================================================================
def test_turn_roundtrip_and_order(store):
    sid = f"{_PREFIX}-turns"
    assert store.save_turn(sid, "第一问", "第一答", ["a.txt"], "answered", 0.9) == 1
    assert store.save_turn(sid, "第二问", "第二答", ["b.txt"], "answered", 0.8) == 2
    assert store.save_turn(sid, "第三问", "第三答") == 3

    turns = store.load_recent_turns(sid, limit=2)
    assert [t["turn_no"] for t in turns] == [2, 3]        # 取最近两轮，且正序
    assert turns[-1]["question"] == "第三问"
    assert turns[0]["answer"] == "第二答"

    full = store.load_recent_turns(sid, limit=10)
    assert [t["turn_no"] for t in full] == [1, 2, 3]
    assert store.count_turns(sid) == 3


def test_save_turn_ignores_empty_answer(store):
    sid = f"{_PREFIX}-empty"
    assert store.save_turn(sid, "只说了问题", "") == 0


# =====================================================================
# 3. 会话实体记忆
# =====================================================================
def test_entities_roundtrip(store):
    sid = f"{_PREFIX}-ents"
    store.ensure_session(sid, "u-test")
    store.save_entities(sid, ["代码", "小李"])
    assert store.load_entities(sid) == ["代码", "小李"]
    store.save_entities(sid, ["布丁"])
    assert store.load_entities(sid) == ["布丁"]        # 覆盖而不是累加


def test_entities_unknown_session_returns_empty(store):
    assert store.load_entities(f"{_PREFIX}-no-such") == []


# =====================================================================
# 4. 长期记忆
# =====================================================================
def test_upsert_dedupes_and_updates_value(store):
    uid = f"{_PREFIX}-u1"
    n = store.upsert_user_memories(
        uid, [{"key": "称呼", "value": "小王", "category": "profile", "confidence": 0.9}], "s1"
    )
    assert n == 1
    store.upsert_user_memories(
        uid, [{"key": "称呼", "value": "王工", "category": "profile", "confidence": 0.8}], "s2"
    )
    mems = store.load_user_memories(uid)
    assert len(mems) == 1
    assert mems[0]["value"] == "王工"        # 新值覆盖旧值
    assert mems[0]["hit_count"] == 2         # 被确认两次


def test_load_user_memories_isolated_per_user(store):
    u1, u2 = f"{_PREFIX}-a", f"{_PREFIX}-b"
    store.upsert_user_memories(u1, [{"key": "城市", "value": "深圳"}])
    store.upsert_user_memories(u2, [{"key": "城市", "value": "北京"}])
    assert store.load_user_memories(u1)[0]["value"] == "深圳"
    assert store.load_user_memories(u2)[0]["value"] == "北京"


def test_load_user_memories_orders_by_hit_count(store):
    uid = f"{_PREFIX}-order"
    store.upsert_user_memories(uid, [{"key": "常问", "value": "川菜"}])
    for _ in range(3):
        store.upsert_user_memories(uid, [{"key": "常问", "value": "川菜"}])
    store.upsert_user_memories(uid, [{"key": "偶问", "value": "粤菜"}])
    mems = store.load_user_memories(uid, limit=5)
    assert mems[0]["key"] == "常问"


def test_forget_one_and_all(store):
    uid = f"{_PREFIX}-forget"
    store.upsert_user_memories(uid, [{"key": "k1", "value": "v1"}, {"key": "k2", "value": "v2"}])
    assert len(store.load_user_memories(uid)) == 2
    assert store.forget_user_memory(uid, "k1") == 1
    assert [m["key"] for m in store.load_user_memories(uid)] == ["k2"]
    assert store.forget_user_memory(uid) == 1
    assert store.load_user_memories(uid) == []


def test_empty_inputs_are_safe(store):
    assert store.upsert_user_memories("", [{"key": "a", "value": "b"}]) == 0
    assert store.upsert_user_memories(f"{_PREFIX}-x", []) == 0
    assert store.load_user_memories("") == []
    assert store.forget_user_memory("") == 0


# =====================================================================
# 5. 降级：数据库不可用时不能把问答拖挂
# =====================================================================
def test_degrades_gracefully_without_db(monkeypatch):
    monkeypatch.setattr(memory_store, "_get_factory", lambda: None)
    assert memory_store.save_turn("s-degrade", "q", "a") == 0
    assert memory_store.load_recent_turns("s-degrade") == []
    assert memory_store.load_entities("s-degrade") == []
    assert memory_store.load_user_memories("u-degrade") == []
    assert memory_store.upsert_user_memories("u-degrade", [{"key": "a", "value": "b"}]) == 0
    assert memory_store.stats()["turns"] == 0
