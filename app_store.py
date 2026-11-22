# -*- coding: utf-8 -*-
"""app_store.py —— 用户账号与「用户可见的」对话归档（SQLite）。

为什么另起一张库，不复用 memory_store 的 chat_turns
──────────────────────────────────────────────────
`chat_turns` 是**给模型读的上下文**，不是给用户看的历史。两者需求正好相反：

| | chat_turns（记忆层） | messages（归档层） |
|---|---|---|
| 生命周期 | `purge_old_turns(days=30)` 会清掉 | 永久保留，用户自己不删就不删 |
| 主键语义 | `(session_id, turn_no)`，轮次是连续的 | 单条消息，允许删除/回填 |
| 用途 | 重建最近 N 轮喂给模型 | 打开会话时原样回放 |
| 依赖 | MySQL（Docker 里那套） | SQLite（clone 下来就能跑） |

所以合并成一张表是错的：一清理就把用户的历史删了；反过来把归档塞进 MySQL，
则 clone 项目的人没有 MySQL 就看不到任何历史——而"登录后能看到自己的对话"
恰恰是这次要交付的东西。

两套各写一份的代价是同一轮问答存两次。接受这个代价，因为职责边界清楚：
**记忆层可以丢，归档层不能丢。**

三条纪律
────────
1. **绝不冒泡**：数据库坏了只能让"这次没有历史"，不能让问答失败。
   所以对外函数一律 try/except 返回安全默认值（写失败也要把回答给用户）。
2. **SQLite 用 WAL**：流式响应是长事务场景，默认 rollback journal 会让
   读请求在写事务期间直接 `database is locked`。
3. **每次操作独立连接**：FastAPI 的同步路由跑在线程池里，连接跨线程共享
   容易踩 `check_same_thread`；短连接 + WAL 的开销在单机场景可以忽略。
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "app.db"

# 环境变量优先，测试里传临时路径用它覆盖
DB_PATH = Path(os.getenv("RAG_APP_DB") or DEFAULT_DB)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    username    TEXT NOT NULL UNIQUE,
    email       TEXT,
    pwd_hash    TEXT NOT NULL,
    created_at  REAL NOT NULL,
    last_login  REAL
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '新对话',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    msg_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    sources         TEXT,
    contexts        TEXT,
    confidence      REAL,
    status          TEXT,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    kind      TEXT NOT NULL,
    username  TEXT,
    ip        TEXT,
    device    TEXT,
    ua        TEXT
);

CREATE INDEX IF NOT EXISTS idx_conv_user  ON conversations(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_msg_conv   ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_event_id   ON auth_events(id);
"""


# =====================================================================
# 连接
# =====================================================================
@contextmanager
def _conn():
    """一次操作一条连接。WAL + busy_timeout 是并发下的两个必需品。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


_initialized = {"done": False}


def init_db() -> None:
    """建表。幂等，进程内只跑一次；失败不抛，让调用方降级。"""
    if _initialized["done"]:
        return
    try:
        with _conn() as conn:
            conn.executescript(_SCHEMA)
        _initialized["done"] = True
    except Exception:  # noqa: BLE001
        # 建表失败不该让服务起不来，第一次真正写入时还会再试
        pass


def _now() -> float:
    return time.time()


def _row(r: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    return dict(r) if r is not None else None


# =====================================================================
# 用户
# =====================================================================
def create_user(username: str, pwd_hash: str, email: str = "") -> Dict[str, Any]:
    """建用户。用户名重复抛 ValueError（调用方转 409）。

    唯一性交给数据库的 UNIQUE 约束，不先 SELECT 再 INSERT：
    两个请求同时注册同一个名字时，"查了没有"这个结论会同时成立，
    于是两边都插进去——这是典型的 check-then-act 竞态。
    """
    init_db()
    uid = uuid.uuid4().hex
    now = _now()
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO users(id, username, email, pwd_hash, created_at) "
                "VALUES(?,?,?,?,?)",
                (uid, username, email or None, pwd_hash, now),
            )
    except sqlite3.IntegrityError:
        raise ValueError("用户名已被占用") from None
    return {"id": uid, "username": username, "email": email or None,
            "created_at": now, "last_login": None}


def get_user_by_name(username: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _conn() as conn:
        return _row(conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone())


def get_user(uid: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _conn() as conn:
        return _row(conn.execute(
            "SELECT * FROM users WHERE id = ?", (uid,)
        ).fetchone())


def touch_login(uid: str) -> None:
    try:
        with _conn() as conn:
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (_now(), uid))
    except Exception:  # noqa: BLE001
        pass


# =====================================================================
# 会话
# =====================================================================
def create_conversation(user_id: str, title: str = "新对话") -> Dict[str, Any]:
    init_db()
    cid = uuid.uuid4().hex
    now = _now()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO conversations(id, user_id, title, created_at, updated_at) "
            "VALUES(?,?,?,?,?)",
            (cid, user_id, (title or "新对话")[:60], now, now),
        )
    return {"id": cid, "user_id": user_id, "title": (title or "新对话")[:60],
            "created_at": now, "updated_at": now, "msg_count": 0}


def list_conversations(user_id: str) -> List[Dict[str, Any]]:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at, msg_count "
            "FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(cid: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _conn() as conn:
        return _row(conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (cid,)
        ).fetchone())


def own_conversation(cid: str, user_id: str) -> Optional[Dict[str, Any]]:
    """取会话并校验归属。**不存在的和别人的返回同一个 None**，
    这样接口不会通过"404 还是 403"泄漏别人会话是否存在。
    """
    c = get_conversation(cid)
    if not c or c["user_id"] != user_id:
        return None
    return c


def rename_conversation(cid: str, user_id: str, title: str) -> bool:
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            ((title or "新对话")[:60], _now(), cid, user_id),
        )
        return cur.rowcount > 0


def delete_conversation(cid: str, user_id: str) -> bool:
    """删会话连同其消息。先校验归属，避免删掉别人的。"""
    with _conn() as conn:
        own = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?", (cid, user_id)
        ).fetchone()
        if not own:
            return False
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
        return True


def touch_conversation(cid: str, title_if_new: str = "") -> None:
    """消息落库后刷新排序时间；标题还是默认值时用首句补上。"""
    try:
        with _conn() as conn:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), cid)
            )
            if title_if_new:
                conn.execute(
                    "UPDATE conversations SET title = ? "
                    "WHERE id = ? AND (title IS NULL OR title = '' OR title = '新对话')",
                    (title_if_new[:60], cid),
                )
    except Exception:  # noqa: BLE001
        pass


# =====================================================================
# 消息
# =====================================================================
def append_message(cid: str, role: str, content: str,
                   sources: Optional[List[str]] = None,
                   contexts: Optional[List[str]] = None,
                   confidence: Optional[float] = None,
                   status: str = "") -> None:
    """写一条消息。**失败只打日志不抛**——回答已经流给用户了，落库失败不能变成 500。"""
    import json as _json
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT INTO messages(conversation_id, role, content, sources, contexts,"
                " confidence, status, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (cid, role, content,
                 _json.dumps(sources or [], ensure_ascii=False) if sources else None,
                 _json.dumps(contexts or [], ensure_ascii=False) if contexts else None,
                 confidence, status or None, _now()),
            )
            conn.execute(
                "UPDATE conversations SET msg_count = msg_count + 1, updated_at = ? "
                "WHERE id = ?", (_now(), cid),
            )
    except Exception:  # noqa: BLE001
        pass


def list_messages(cid: str, limit: int = 500) -> List[Dict[str, Any]]:
    import json as _json
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, sources, contexts, confidence, status, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY id ASC LIMIT ?",
            (cid, max(1, min(limit, 2000))),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("sources", "contexts"):
            if d.get(k):
                try:
                    d[k] = _json.loads(d[k])
                except Exception:  # noqa: BLE001
                    d[k] = []
            else:
                d[k] = []
        out.append(d)
    return out


# =====================================================================
# 鉴权事件（给管理员推送"有人来了"）
# =====================================================================
def log_event(kind: str, username: str = "", ip: str = "",
              device: str = "", ua: str = "") -> int:
    """记一条注册/登录事件，返回自增 id（0 表示没记上）。

    为什么单独开表、而不是让通知程序去看 `users.last_login`：
    推送需要一个**单调递增的游标**。时间戳做游标会在同一秒并发时漏事件
    （`>` 比较直接跳过同值行），也会被客户端时钟漂移影响；
    AUTOINCREMENT 的 id 天然有序、天然能断点续传。

    失败只返回 0，绝不抛出：记日志失败不该让登录失败。
    """
    try:
        init_db()
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO auth_events(ts, kind, username, ip, device, ua) "
                "VALUES(?,?,?,?,?,?)",
                (_now(), kind, username or None, ip or None,
                 device or None, (ua or "")[:300] or None),
            )
            return int(cur.lastrowid or 0)
    except Exception:  # noqa: BLE001
        return 0


def list_events(since_id: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
    """取 id 大于 since_id 的事件，按 id 升序。本机用它做增量拉取。"""
    try:
        init_db()
        with _conn() as conn:
            rows = conn.execute(
                "SELECT id, ts, kind, username, ip, device, ua FROM auth_events "
                "WHERE id > ? ORDER BY id ASC LIMIT ?",
                (max(0, int(since_id)), max(1, min(int(limit), 200))),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:  # noqa: BLE001
        return []


def latest_event_id() -> int:
    """当前最大 id。首次启动时用它对齐游标，避免把历史事件当新事件弹一遍。"""
    try:
        init_db()
        with _conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM auth_events").fetchone()
        return int(row[0] or 0)
    except Exception:  # noqa: BLE001
        return 0


def stats() -> Dict[str, int]:
    """给 /health 用：一眼看出库里有没有东西，而不是只看 status=ok。"""
    try:
        init_db()
        with _conn() as conn:
            return {
                "users": conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                "conversations": conn.execute(
                    "SELECT COUNT(*) FROM conversations").fetchone()[0],
                "messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                "events": conn.execute("SELECT COUNT(*) FROM auth_events").fetchone()[0],
            }
    except Exception:  # noqa: BLE001
        return {"users": 0, "conversations": 0, "messages": 0, "events": 0}
