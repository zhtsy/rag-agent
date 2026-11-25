"""Neo4j 连接层（惰性）。

这个文件原来还负责「自然语言转 Cypher 查图」，那套 demo 图谱已于 2026-11-25 下线
（730 个实体里零个法律概念，法律题在实体探测那一步恒判否，整条链一次都不执行）。
现在只剩连接：`graph.query(...)` 的写法保持不变，真正用到时才建 driver。
法律侧的图查询在 law_graph.py，复用这里的同一个 `graph` 对象。

为什么连接必须惰性（2026-11-05 两次演练）：原来在模块顶层直接 Neo4jGraph(...)，
于是 import 本模块就等于「必须连上 Neo4j」。实测停掉 neo4j 后
`docker compose restart backend` 直接失败 —— Neo4j 挂了，连不需要图谱的普通问答
也起不来，而且进入「Neo4j 没好 → 后端起不来 → 没人能修」的死循环。
现在首次真正用到才连：Neo4j 缺席只影响图谱那一路，向量+BM25 照常服务，
恢复后 15 秒内自动接上。

超时压到 3 秒也不是调优项，是可用性：driver 默认的事务重试要跑满 ~98 秒才抛异常，
这期间 HTTP 是挂住的 —— 不报错、答案最终也对，但用户等 100 秒，观感就是「服务死了」，
比直接 500 更难排查。
"""

import threading
import time
from typing import Any

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from loguru import logger

load_dotenv()

# 连接参数统一由 settings.py 提供（这里不再各写一份默认值，
# 否则换口令时要一家一家翻文件——历史上就是这么漏掉 build_graph.py 的）
from settings import neo4j_kwargs  # noqa: E402


class GraphUnavailable(RuntimeError):
    """Neo4j 连不上。

    单独成一个类型，是为了跟"查询本身写错了"区分开：后者值得重试，
    前者重试只是把同一个超时再等一遍 —— 降级演练实测，无效重试让 graph
    阶段从 5s 涨到 11s。
    """


# 连接超时与事务重试都压到 3 秒：不可用时 3~6 秒内失败并降级，
# 而正常查询是毫秒级，3 秒对这个库的查询量绰绰有余，不会误伤。
_GRAPH_RETRY_COOLDOWN = 15.0  # 连不上后的冷却：别让每个请求都白等 3 秒

_graph: Any = None
_graph_lock = threading.Lock()
_graph_failed_at = 0.0


def _build_graph() -> Neo4jGraph:
    return Neo4jGraph(
        timeout=3,
        driver_config={
            "connection_timeout": 3,
            "connection_acquisition_timeout": 5,
            "max_transaction_retry_time": 3,
        },
        **neo4j_kwargs(),
    )


def get_graph() -> Neo4jGraph:
    """取图谱连接（首次调用才建）。连不上抛 GraphUnavailable，由调用方降级。"""
    global _graph, _graph_failed_at

    if _graph is not None:
        return _graph

    if time.monotonic() - _graph_failed_at < _GRAPH_RETRY_COOLDOWN:
        raise GraphUnavailable("Neo4j 不可用（冷却中，跳过重连）")

    with _graph_lock:  # 并发首问只建一个连接，别 4 个请求建 4 个 driver
        if _graph is None:
            try:
                _graph = _build_graph()
                _graph_failed_at = 0.0
                logger.info("✅ Neo4j 连接已建立（惰性连接）")
            except Exception as e:  # noqa: BLE001
                _graph_failed_at = time.monotonic()
                raise GraphUnavailable(
                    f"Neo4j 连接失败: {type(e).__name__}: {e}"
                ) from e
    return _graph


class _LazyGraph:
    """保持下面 `graph.query(...)` / `graph.schema` 的写法不变，用到时才连库。

    不直接把 `graph` 换成 `get_graph()` 调用，是因为它有 4 个使用点，
    改调用点就得逐个确认 try 覆盖范围；代理对象让"没连上"只在用到处暴露。
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(get_graph(), name)


graph = _LazyGraph()
