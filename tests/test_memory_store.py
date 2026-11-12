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

# CI 只装 pytest（见 .github/workflows/ci.yml），不装 sqlalchemy。
# 模块级跳过而不是让整个 collection 报错：这个文件本来就是"连不上库就 skip"的语义，
# 依赖缺失等价于"环境不具备"，不该把整条 CI 拖红（一红之后就没人看了）。
pytest.importorskip("sqlalchemy", reason="memory_store 依赖 sqlalchemy，CI 不装")

import memory_store  # noqa: E402

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
# 4. 滚动摘要
# =====================================================================
def test_summary_roundtrip_and_monotonic_progress(store):
    sid = f"{_PREFIX}-sum1"
    assert store.load_summary(sid)["upto_turn"] == 0        # 还没有摘要

    store.save_summary(sid, "第一版摘要", 3)
    data = store.load_summary(sid)
    assert data["summary"] == "第一版摘要" and data["upto_turn"] == 3

    store.save_summary(sid, "第二版摘要", 6)
    data = store.load_summary(sid)
    assert data["summary"] == "第二版摘要" and data["upto_turn"] == 6

    # upto_turn 只能前进：并发下旧值不能把进度倒退，
    # 否则同一批轮次会被重复摘要，摘要越压越糊。
    store.save_summary(sid, "旧值", 2)
    assert store.load_summary(sid)["upto_turn"] == 6


def test_load_turns_range_boundaries_and_order(store):
    sid = f"{_PREFIX}-sum2"
    for i in range(1, 6):
        store.save_turn(sid, f"问题{i}", f"回答{i}")

    rows = store.load_turns_range(sid, 2, 4)
    assert [r["turn_no"] for r in rows] == [2, 3, 4]        # 闭区间 + 正序
    assert rows[0]["question"] == "问题2"

    assert store.load_turns_range(sid, 4, 2) == []          # 反向区间直接空
    assert store.load_turns_range(sid, 1, 99)[-1]["turn_no"] == 5   # 上界超出不报错


def test_summary_empty_inputs_are_safe(store):
    assert store.load_summary("")["upto_turn"] == 0
    store.save_summary("", "x", 1)                          # 空 session 不炸
    store.save_summary(f"{_PREFIX}-s", "", 1)               # 空摘要不写
    assert store.load_summary(f"{_PREFIX}-s")["summary"] == ""
    assert store.load_turns_range("", 1, 2) == []


# =====================================================================
# 5. 降级：数据库不可用时不能把问答拖挂
# =====================================================================
def test_degrades_gracefully_without_db(monkeypatch):
    monkeypatch.setattr(memory_store, "_get_factory", lambda: None)
    assert memory_store.save_turn("s-degrade", "q", "a") == 0
    assert memory_store.load_recent_turns("s-degrade") == []
    assert memory_store.load_entities("s-degrade") == []
    assert memory_store.load_summary("s-degrade") == {"summary": "", "upto_turn": 0}
    assert memory_store.load_turns_range("s-degrade", 1, 3) == []
    memory_store.save_summary("s-degrade", "x", 1)           # 降级路径下不抛异常
    assert memory_store.stats()["turns"] == 0
