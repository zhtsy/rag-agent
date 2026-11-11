# -*- coding: utf-8 -*-
"""对话记忆的持久化层（MySQL）。

为什么不换一个"持久化 checkpointer"（SqliteSaver / PostgresSaver）就完事
──────────────────────────────────────────────────────────────
`MemorySaver` 的毛病是"只活在这个进程里"，看起来换个落盘 checkpointer 就解决了。
但它只解决"单机重启"，解决不了"同一个用户的下一句话被负载均衡到另一个 worker"：
sqlite 文件是容器内单点的，第二个 worker 读不到；而 PostgresSaver 要再引一套依赖。
线上是 docker 五服务 + 可能多副本，所以真正需要的是**落在一张所有 worker 都看得见的表里**。

于是这里只存业务层真正需要的三样东西，图状态在进程内按需重建：

    chat_sessions   会话元数据 + 会话级实体（代词消解用），重启后能恢复
    chat_turns      每一轮的完整问答（不截断），重启/换 worker 时回填最近 N 轮
    user_memories   用户级长期事实，跨会话、跨设备可用

三条纪律（比功能本身重要）
──────────────────────────
1. **不许冒泡**：数据库挂了要降级成"这次没有记忆"，绝不能因此让问答失败。
   所以每个函数都 try/except，返回安全默认值，并按 60 秒退避重试，避免每轮都打一次挂掉的库。
2. **评测会话不落库**：session_id 以 `eval-` 开头默认跳过。
   否则同一批题第二次跑会带着第一次的历史，分数会虚高——这类"假通过"最难发现。
3. **建表在首次调用时自动完成**：不依赖迁移脚本，新环境直接能跑。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.orm import sessionmaker

try:  # 统一走项目日志；测试环境没有 logs/ 目录时退回标准库
    from logger_config import logger  # type: ignore
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger(__name__)


# =====================================================================
# 0. 开关与降级
# =====================================================================
ENABLED = (os.getenv("RAG_MEMORY_STORE", "1").strip().lower() not in {"0", "false", "off", "no"})
# 评测会话是否也落库。默认 off——理由见文件头纪律 2。
STORE_EVAL_SESSIONS = (
    os.getenv("RAG_MEMORY_EVAL_SESSIONS", "0").strip().lower() in {"1", "true", "on", "yes"}
)
EVAL_SESSION_PREFIXES = ("eval-", "eval_", "ragas-", "test_eval")

_RETRY_BACKOFF_S = 60.0
_last_fail_at = 0.0
_engine = None
_session_factory = None
_tables_ready = False


def enabled_for(session_id: str) -> bool:
    """这个会话要不要用持久化记忆。

    除了总开关，还要拦评测会话：评测必须每题都是"干净开局"，
    一旦吃到上一轮的历史，召回率和拒答判定全都会被污染。
    """
    if not ENABLED:
        return False
    sid = (session_id or "").strip().lower()
    if not STORE_EVAL_SESSIONS and sid.startswith(EVAL_SESSION_PREFIXES):
        return False
    return True


def _db_url() -> Optional[str]:
    try:
        from settings import DATABASE_URL

        return DATABASE_URL or None
    except Exception:  # noqa: BLE001
        return None


def _get_factory():
    """惰性建 engine + 建表。失败后 60 秒内不再重试，避免拖慢每一轮问答。"""
    global _engine, _session_factory, _tables_ready, _last_fail_at

    if _session_factory is not None and _tables_ready:
        return _session_factory
    if not ENABLED:
        return None
    if _last_fail_at and (time.time() - _last_fail_at) < _RETRY_BACKOFF_S:
        return None

    url = _db_url()
    if not url:
        logger.warning("🧠 记忆存储不可用：未配置 DATABASE_URL，本轮按无记忆处理")
        _last_fail_at = time.time()
        return None

    try:
        _engine = create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True, pool_recycle=1800)
        _session_factory = sessionmaker(bind=_engine)
        _create_tables(_session_factory)
        _tables_ready = True
        _last_fail_at = 0.0
        return _session_factory
    except Exception as e:  # noqa: BLE001
        logger.warning(f"🧠 记忆存储初始化失败（降级为无记忆）: {type(e).__name__}: {e}")
        _session_factory = None
        _tables_ready = False
        _last_fail_at = time.time()
        return None


@contextmanager
def _conn():
    """拿一个 session；拿不到就 yield None，调用方据此短路。

    这里**故意把异常吞掉**：记忆是增强功能，数据库抖一下不该让用户这轮问不出话。
    代价是如果 SQL 写错了也会被吞，所以失败时一定要打 warning 并且进入退避
    （60 秒内不再碰库），否则会变成"每轮都静默失败、日志刷屏"。
    """
    global _last_fail_at

    factory = _get_factory()
    if factory is None:
        yield None
        return
    session = factory()
    try:
        yield session
        session.commit()
    except Exception as e:  # noqa: BLE001
        _last_fail_at = time.time()
        try:
            session.rollback()
        except Exception:  # noqa: BLE001
            pass
        logger.warning(f"🧠 记忆存储操作失败（已忽略，60s 内降级）: {type(e).__name__}: {e}")
    finally:
        session.close()


_TABLE_DDL = (
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id VARCHAR(128) NOT NULL PRIMARY KEY,
        user_id VARCHAR(128) NULL,
        entities TEXT NULL,
        turn_count INT NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_session_user (user_id, updated_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_turns (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(128) NOT NULL,
        turn_no INT NOT NULL,
        question LONGTEXT NOT NULL,
        answer LONGTEXT NOT NULL,
        sources LONGTEXT NULL,
        status VARCHAR(32) NULL,
        confidence DOUBLE NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_session_turn (session_id, turn_no),
        INDEX idx_session_created (session_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS user_memories (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        user_id VARCHAR(128) NOT NULL,
        mem_key VARCHAR(64) NOT NULL,
        mem_value VARCHAR(255) NOT NULL,
        category VARCHAR(32) NOT NULL DEFAULT 'profile',
        confidence DOUBLE NOT NULL DEFAULT 0.6,
        hit_count INT NOT NULL DEFAULT 1,
        source_session VARCHAR(128) NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_hit_at DATETIME NULL,
        UNIQUE KEY uk_user_key (user_id, mem_key),
        INDEX idx_user_updated (user_id, updated_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
)


def _create_tables(factory) -> None:
    with factory() as session:
        for ddl in _TABLE_DDL:
            session.execute(text(ddl))
        session.commit()


# =====================================================================
# 1. 会话与实体记忆
# =====================================================================
def ensure_session(session_id: str, user_id: str = "") -> None:
    """登记会话（幂等）。user_id 只在第一次写入，之后不覆盖，
    否则同一浏览器换了用户名，历史 turn 的归属会跟着漂移。"""
    if not session_id or not enabled_for(session_id):
        return
    with _conn() as s:
        if s is None:
            return
        s.execute(
            text(
                """INSERT INTO chat_sessions (session_id, user_id)
                   VALUES (:sid, :uid)
                   ON DUPLICATE KEY UPDATE
                     user_id = COALESCE(user_id, VALUES(user_id))"""
            ),
            {"sid": session_id, "uid": (user_id or None)},
        )


def save_entities(session_id: str, entities: Sequence[str]) -> None:
    """把会话级实体写下来。

    这份东西是代词消解的最后一道保险：重启后 `_ENTITY_MEMORY` 是空的，
    用户再说"他住在哪"，没有它就一定答不出来。
    """
    if not session_id or not enabled_for(session_id):
        return
    payload = json.dumps(list(entities or [])[:20], ensure_ascii=False)
    with _conn() as s:
        if s is None:
            return
        s.execute(
            text(
                """INSERT INTO chat_sessions (session_id, entities)
                   VALUES (:sid, :ents)
                   ON DUPLICATE KEY UPDATE entities = VALUES(entities)"""
            ),
            {"sid": session_id, "ents": payload},
        )


def load_entities(session_id: str) -> List[str]:
    if not session_id:
        return []
    with _conn() as s:
        if s is None:
            return []
        row = s.execute(
            text("SELECT entities FROM chat_sessions WHERE session_id = :sid"),
            {"sid": session_id},
        ).fetchone()
    if not row or not row[0]:
        return []
    try:
        data = json.loads(row[0])
        return [str(x) for x in data][:20] if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


# =====================================================================
# 2. 每一轮问答（短期记忆的持久化载体）
# =====================================================================
def save_turn(
    session_id: str,
    question: str,
    answer: str,
    sources: Optional[Iterable[str]] = None,
    status: str = "",
    confidence: Optional[float] = None,
) -> int:
    """存一轮完整问答，返回 turn_no（0 表示没存成）。

    存**完整答案**而不是压缩后的版本：压缩是"给模型看"时才做的事，
    原始记录留着才可能在重启后重新压缩、或者换一种窗口策略重放。
    """
    if not session_id or not answer or not enabled_for(session_id):
        return 0
    with _conn() as s:
        if s is None:
            return 0
        row = s.execute(
            text("SELECT COALESCE(MAX(turn_no), 0) FROM chat_turns WHERE session_id = :sid"),
            {"sid": session_id},
        ).fetchone()
        turn_no = int(row[0] or 0) + 1
        s.execute(
            text(
                """INSERT INTO chat_turns
                   (session_id, turn_no, question, answer, sources, status, confidence)
                   VALUES (:sid, :no, :q, :a, :src, :st, :cf)"""
            ),
            {
                "sid": session_id,
                "no": turn_no,
                "q": question or "",
                "a": answer,
                "src": json.dumps(list(sources or []), ensure_ascii=False),
                "st": (status or "")[:32],
                "cf": confidence,
            },
        )
        s.execute(
            text(
                """INSERT INTO chat_sessions (session_id, turn_count)
                   VALUES (:sid, 1)
                   ON DUPLICATE KEY UPDATE turn_count = turn_count + 1"""
            ),
            {"sid": session_id},
        )
        return turn_no


def load_recent_turns(session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    """取最近 limit 轮，按时间正序返回（最老的在前，方便直接拼成消息序列）。"""
    if not session_id or limit <= 0:
        return []
    with _conn() as s:
        if s is None:
            return []
        rows = s.execute(
            text(
                """SELECT turn_no, question, answer, status, confidence, created_at
                   FROM chat_turns WHERE session_id = :sid
                   ORDER BY turn_no DESC LIMIT :lim"""
            ),
            {"sid": session_id, "lim": int(limit)},
        ).fetchall()
    out = [
        {
            "turn_no": r[0],
            "question": r[1] or "",
            "answer": r[2] or "",
            "status": r[3] or "",
            "confidence": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]
    out.reverse()
    return out


def count_turns(session_id: str) -> int:
    if not session_id:
        return 0
    with _conn() as s:
        if s is None:
            return 0
        return int(s.execute(
            text("SELECT COUNT(*) FROM chat_turns WHERE session_id = :sid"),
            {"sid": session_id},
        ).scalar() or 0)


# =====================================================================
# 3. 用户级长期记忆
# =====================================================================
def upsert_user_memories(
    user_id: str,
    facts: Sequence[Dict[str, Any]],
    source_session: str = "",
) -> int:
    """写入/更新用户事实，返回实际写入条数。

    去重键是 (user_id, mem_key)：同一件事第二次被抽到，
    不新增一行，而是**覆盖 value 并把 hit_count +1**。
    这样"去过上海"改成"搬到深圳"能自动更新，
    而反复出现的事实会自然排到前面（读取时按 hit_count 排序）。
    """
    if not user_id or not facts:
        return 0
    written = 0
    with _conn() as s:
        if s is None:
            return 0
        now = datetime.now()
        for f in facts:
            key = str(f.get("key") or "").strip()[:64]
            value = str(f.get("value") or "").strip()[:255]
            if not key or not value:
                continue
            s.execute(
                text(
                    """INSERT INTO user_memories
                       (user_id, mem_key, mem_value, category, confidence, source_session, updated_at)
                       VALUES (:uid, :k, :v, :cat, :cf, :src, :now)
                       ON DUPLICATE KEY UPDATE
                         mem_value = VALUES(mem_value),
                         category = VALUES(category),
                         confidence = GREATEST(confidence, VALUES(confidence)),
                         hit_count = hit_count + 1,
                         source_session = VALUES(source_session),
                         updated_at = VALUES(updated_at)"""
                ),
                {
                    "uid": user_id,
                    "k": key,
                    "v": value,
                    "cat": str(f.get("category") or "profile")[:32],
                    "cf": float(f.get("confidence") or 0.6),
                    "src": (source_session or "")[:128] or None,
                    "now": now,
                },
            )
            written += 1
    return written


def load_user_memories(
    user_id: str,
    limit: int = 8,
    days: int = 180,
) -> List[Dict[str, Any]]:
    """按"被确认次数 + 新鲜度"取用户事实。

    days 是**软遗忘**：太久没被再次确认的事实不再注入，但也不删。
    "永远不会忘"听起来是优点，实际是灾难——用户三个月前说过的话还挂在 prompt 里，
    模型会把它当成当前状态（"你上次说在准备考研"）。
    """
    if not user_id or limit <= 0:
        return []
    cutoff = datetime.now() - timedelta(days=max(1, days))
    with _conn() as s:
        if s is None:
            return []
        rows = s.execute(
            text(
                """SELECT mem_key, mem_value, category, confidence, hit_count, updated_at
                   FROM user_memories
                   WHERE user_id = :uid AND updated_at >= :cutoff
                   ORDER BY hit_count DESC, updated_at DESC
                   LIMIT :lim"""
            ),
            {"uid": user_id, "cutoff": cutoff, "lim": int(limit)},
        ).fetchall()
        if not rows:
            return []
        # 顺手记一次"被用到"，用于以后做衰减/清理
        keys = [r[0] for r in rows]
        s.execute(
            text(
                "UPDATE user_memories SET last_hit_at = :now "
                "WHERE user_id = :uid AND mem_key IN :keys"
            ).bindparams(bindparam("keys", expanding=True)),
            {"now": datetime.now(), "uid": user_id, "keys": keys},
        )
    return [
        {
            "key": r[0],
            "value": r[1],
            "category": r[2],
            "confidence": r[3],
            "hit_count": r[4],
            "updated_at": r[5],
        }
        for r in rows
    ]


def forget_user_memory(user_id: str, key: Optional[str] = None) -> int:
    """删掉某个用户的一条或全部记忆，返回删除条数。"""
    if not user_id:
        return 0
    with _conn() as s:
        if s is None:
            return 0
        if key:
            res = s.execute(
                text("DELETE FROM user_memories WHERE user_id = :uid AND mem_key = :k"),
                {"uid": user_id, "k": key},
            )
        else:
            res = s.execute(
                text("DELETE FROM user_memories WHERE user_id = :uid"),
                {"uid": user_id},
            )
        return int(res.rowcount or 0)


# =====================================================================
# 4. 清理与统计
# =====================================================================
def purge_old_turns(days: int = 30, dry_run: bool = True) -> Dict[str, int]:
    """清理过老的会话轮次。长期事实不跟着删——它们本来就该比会话活得久。"""
    cutoff = datetime.now() - timedelta(days=max(1, days))
    with _conn() as s:
        if s is None:
            return {"deleted": 0, "remaining": 0}
        n = int(s.execute(
            text("SELECT COUNT(*) FROM chat_turns WHERE created_at < :c"), {"c": cutoff}
        ).scalar() or 0)
        if not dry_run and n:
            s.execute(text("DELETE FROM chat_turns WHERE created_at < :c"), {"c": cutoff})
        remaining = int(s.execute(text("SELECT COUNT(*) FROM chat_turns")).scalar() or 0)
    return {"deleted": 0 if dry_run else n, "would_delete": n, "remaining": remaining}


def stats() -> Dict[str, int]:
    with _conn() as s:
        if s is None:
            return {"sessions": 0, "turns": 0, "memories": 0, "users": 0}
        return {
            "sessions": int(s.execute(text("SELECT COUNT(*) FROM chat_sessions")).scalar() or 0),
            "turns": int(s.execute(text("SELECT COUNT(*) FROM chat_turns")).scalar() or 0),
            "memories": int(s.execute(text("SELECT COUNT(*) FROM user_memories")).scalar() or 0),
            "users": int(s.execute(text("SELECT COUNT(DISTINCT user_id) FROM user_memories")).scalar() or 0),
        }


# =====================================================================
# 5. CLI（演示与运维用；不暴露成 HTTP 接口，避免记忆被未授权读取）
# =====================================================================
if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    if not argv:
        print("用法:")
        print("  python memory_store.py stats")
        print("  python memory_store.py show <user_id>")
        print("  python memory_store.py forget <user_id> [key]")
        print("  python memory_store.py turns <session_id> [limit]")
        print("  python memory_store.py purge [days] --run")
        sys.exit(0)

    cmd = argv[0]
    if cmd == "stats":
        print(stats())
    elif cmd == "show" and len(argv) > 1:
        for m in load_user_memories(argv[1], limit=50, days=3650):
            print(f"[{m['category']}] {m['key']}: {m['value']}  (hit={m['hit_count']}, conf={m['confidence']})")
    elif cmd == "forget" and len(argv) > 1:
        key = argv[2] if len(argv) > 2 else None
        print(f"已删除 {forget_user_memory(argv[1], key)} 条")
    elif cmd == "turns" and len(argv) > 1:
        lim = int(argv[2]) if len(argv) > 2 else 5
        for t in load_recent_turns(argv[1], limit=lim):
            print(f"#{t['turn_no']} Q: {t['question']}\n     A: {t['answer'][:60]}...")
    elif cmd == "purge":
        days = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 30
        print(purge_old_turns(days, dry_run="--run" not in argv))
    else:
        print(f"未知命令: {cmd}")
