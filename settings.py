# -*- coding: utf-8 -*-
"""settings.py —— 全部外部依赖与可调参数的**唯一读取点**。

为什么要有它
────────────
同一个 Neo4j 口令原来分散在 4 个地方（graph_retriever.py / build_graph.py /
fix_source.py / docker-compose.yml），其中两处还是写死的字面量；MySQL 口令更是
直接以默认值的形式躺在 admin_app.py 里。后果有两个，都真实发生过：

1. **改一处忘一处**：换成新口令后 build_graph.py 连不上库，报的却是
   "AuthenticationRateLimit"，要一家一家翻文件才知道漏了谁。
2. **密钥进 git**：字面量默认值等于把生产口令写进版本历史，删都删不干净。

现在统一成：**代码只认环境变量，字面量只出现在 .env（不进 git）**。
`docker-compose.yml` 仍然会给容器注入环境变量——那是部署清单，不是代码。

用法
────
    from settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, neo4j_kwargs
    graph = Neo4jGraph(**neo4j_kwargs())

    from settings import require_database_url
    engine = create_engine(require_database_url())

关于默认值
──────────
非敏感项（bolt 地址、端口、阈值）保留默认值，方便本地直接跑。
口令类一律**不提供默认值**：`require_*()` / `neo4j_kwargs()` 在缺失时抛出
带修复指引的异常，比"静默连一个错的地址、然后报个看不懂的错"好得多，
更重要的是不会把生产口令写进 git。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 允许 .env 覆盖（本地开发）；容器里由 compose 的 environment 注入，不依赖 .env
load_dotenv(Path(__file__).with_name(".env"))

# 记录了"哪些敏感项正在用默认值"，供 warn_insecure_defaults() 打告警
_INSECURE_DEFAULTS: list = []


def _env(key: str, default=None, *, secret: bool = False) -> str:
    """读环境变量。空字符串按"没配"处理（compose 的 ${VAR:-} 会给空串）。"""
    value = os.environ.get(key)
    if value is not None and value.strip() != "":
        return value.strip()
    if secret and default:
        _INSECURE_DEFAULTS.append(key)
    return default


def _env_num(key: str, default, cast=float):
    raw = _env(key)
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def _env_flag(key: str, default: bool = True) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


# ========== Neo4j ==========
NEO4J_URI = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = _env("NEO4J_USER", "neo4j")
# 口令不设默认值：字面量默认值等于把口令写进 git 历史，删都删不干净。
NEO4J_PASSWORD = _env("NEO4J_PASSWORD")


def neo4j_kwargs() -> dict:
    """给 Neo4jGraph(...) 用的参数字典，避免每个文件各拼一遍。"""
    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "未配置 NEO4J_PASSWORD。\n"
            "  · 容器里：由 docker-compose 的 environment 注入（NEO4J_PASSWORD=...）\n"
            "  · 本地跑：cp .env.example .env 并填上 NEO4J_PASSWORD\n"
            "这里刻意不给默认口令——写死默认值会把生产密码提交进 git。"
        )
    return {"url": NEO4J_URI, "username": NEO4J_USER, "password": NEO4J_PASSWORD}


# ========== MySQL（问答记录 + 后台） ==========
DATABASE_URL = _env("DATABASE_URL")


def require_database_url() -> str:
    if not DATABASE_URL:
        raise RuntimeError(
            "未配置 DATABASE_URL。\n"
            "  · 容器里：由 docker-compose 的 environment 注入，检查 service 是否配了\n"
            "  · 本地跑：cp .env.example .env 并填上 mysql+pymysql://user:pass@host:3306/rag_qa\n"
            "不提供默认值是有意的——写死默认口令等于把生产库密码提交进 git。"
        )
    return DATABASE_URL


# ========== 模型 ==========
OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# DeepSeek 的 key 由 langchain_deepseek 直接读环境变量，这里只做存在性检查
DEEPSEEK_API_KEY = _env("DEEPSEEK_API_KEY")


# ========== RAG 行为开关（与 rag_agent.py 里的读取保持一致） ==========
# 检索相关性门槛：rerank 原始 top1 低于它就判"知识库无相关内容"（实测 OOD ≤0.05、正常题 ≥0.6）
MIN_RELEVANT = _env_num("RAG_MIN_RELEVANT", 0.15)
# rerank 窗口：太小会截掉高分块（曾经设 6 直接把 T019 答废），见 rag_agent 注释
RERANK_CANDIDATES = _env_num("RAG_RERANK_CANDIDATES", 10, int)
# 完整性兜底总开关（answer_guard.py）
ANSWER_GUARD = _env_flag("RAG_ANSWER_GUARD", True)
# 请求级耗时日志（obs.py）
TRACE_LOG = _env_flag("RAG_TRACE_LOG", True)
TRACE_FILE = _env("RAG_TRACE_FILE", "") or ""


def warn_insecure_defaults(logger=None) -> None:
    """启动时喊一嗓子：哪些敏感配置还在用默认值。只喊一次。"""
    if not _INSECURE_DEFAULTS:
        return
    names = "、".join(sorted(set(_INSECURE_DEFAULTS)))
    msg = (f"⚠️ 以下敏感配置正在使用内置默认值：{names}。"
           f"请写进 .env（不进 git）或 compose 的 environment。")
    if logger is not None:
        logger.warning(msg)
    else:
        print(msg, flush=True)


def describe() -> dict:
    """脱敏后的配置快照，给健康检查/后台展示用（口令一律不出现在这里）。"""
    return {
        "neo4j_uri": NEO4J_URI,
        "neo4j_user": NEO4J_USER,
        "neo4j_password_set": bool(NEO4J_PASSWORD),
        "database_url_set": bool(DATABASE_URL),
        "database_host": (DATABASE_URL.split("@")[-1].split("/")[0]
                          if DATABASE_URL and "@" in DATABASE_URL else "unknown"),
        "ollama": OLLAMA_BASE_URL,
        "deepseek_key_set": bool(DEEPSEEK_API_KEY),
        "min_relevant": MIN_RELEVANT,
        "rerank_candidates": RERANK_CANDIDATES,
        "answer_guard": ANSWER_GUARD,
        "trace_log": TRACE_LOG,
    }
