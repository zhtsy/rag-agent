# -*- coding: utf-8 -*-
"""graph_retriever 的惰性连接：不连库也能 import，连不上要快速降级。

覆盖一次真实故障（2026-11-05）：Neo4j 停机时 backend 起不来。
根因是模块级 `graph = Neo4jGraph(...)`——"import 本模块"等于"必须连上库"：

    File "/app/graph_retriever.py", line 54, in <module>
    ValueError: Could not connect to Neo4j database

后果不是"图谱那一路不可用"，而是整个后端（包括不需要图谱的普通问答）
进入重启循环，且在 Neo4j 修好之前无法自愈。这些用例就是钉住这个行为。
"""

import pytest

# 这条测试要 import 真实的 graph_retriever，因此依赖后端全套（langchain / Neo4j driver）。
# 宿主机与 CI 只装了 pytest，跑不了——明确 skip 而不是伪装成通过：
#   - CI（只装 pytest）：跳过，靠 eval_gate 在服务器上守
#   - 容器内（有全套依赖）：真跑
# 用 importorskip 而不是 try/except，是为了让"为什么没跑"在报告里可见。
pytest.importorskip("langchain", reason="需要后端依赖，请在 backend 容器内运行")
pytest.importorskip("langchain_neo4j", reason="需要 Neo4j driver")
pytest.importorskip("loguru", reason="需要后端依赖")

import graph_retriever as gr  # noqa: E402


class _FakeGraph:
    schema = "(:Person {id, name})"

    def __init__(self):
        self.queries = []

    def query(self, cypher):
        self.queries.append(cypher)
        return [{"id": "小张"}]


class _ExplodingLLM:
    """被调用就失败：用来证明"连不上时不该白跑一次 LLM"。"""

    def invoke(self, *_a, **_k):
        raise AssertionError("图谱不可用时不应该调用 LLM 生成 Cypher")


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """每个用例都从"还没连过"开始，避免用例之间互相污染单例。"""
    monkeypatch.setattr(gr, "_graph", None)
    monkeypatch.setattr(gr, "_graph_failed_at", 0.0)


# ─── 1. 惰性：拿到代理不等于连上 ───────────────────────────────
def test_proxy_defers_connection_until_used(monkeypatch):
    calls = []

    def _build():
        calls.append(1)
        return _FakeGraph()

    monkeypatch.setattr(gr, "_build_graph", _build)

    _ = gr.graph          # 只是拿到代理对象
    assert calls == [], "拿到 graph 对象时不应该连库"

    assert gr.graph.schema == _FakeGraph.schema
    assert len(calls) == 1, "第一次真正使用才建连接"


def test_proxy_forwards_query(monkeypatch):
    fake = _FakeGraph()
    monkeypatch.setattr(gr, "_build_graph", lambda: fake)

    rows = gr.graph.query("MATCH (n) RETURN n")
    assert rows == [{"id": "小张"}]
    assert fake.queries == ["MATCH (n) RETURN n"]


# ─── 2. 连不上：抛专用异常，而不是 ValueError / 崩进程 ─────────
def test_connection_failure_raises_graph_unavailable(monkeypatch):
    def _boom():
        raise ValueError("Could not connect to Neo4j database")

    monkeypatch.setattr(gr, "_build_graph", _boom)

    with pytest.raises(gr.GraphUnavailable):
        gr.get_graph()


def test_failure_sets_cooldown_and_skips_reconnect(monkeypatch):
    """失败后冷却期内不再重试：否则每个请求都要白等一次 3 秒连接超时。"""
    calls = []

    def _boom():
        calls.append(1)
        raise ValueError("nope")

    monkeypatch.setattr(gr, "_build_graph", _boom)

    with pytest.raises(gr.GraphUnavailable):
        gr.get_graph()
    with pytest.raises(gr.GraphUnavailable) as exc:
        gr.get_graph()

    assert "冷却" in str(exc.value)
    assert len(calls) == 1, "冷却期内不该再尝试建连接"


def test_success_is_cached(monkeypatch):
    """并发首问只建一个 driver。"""
    calls = []
    monkeypatch.setattr(gr, "_build_graph", lambda: calls.append(1) or _FakeGraph())

    first = gr.get_graph()
    second = gr.get_graph()

    assert first is second
    assert len(calls) == 1


# ─── 3. 失败要短路，不要"带着连不上的报错再改一版 Cypher" ─────────
def test_graph_unavailable_counts_as_fatal():
    assert gr.GraphUnavailable in gr._FATAL_DB_ERRORS


def test_run_cypher_short_circuits_without_calling_llm(monkeypatch):
    def _boom():
        raise ValueError("Could not connect to Neo4j database")

    monkeypatch.setattr(gr, "_build_graph", _boom)
    monkeypatch.setattr(gr, "llm", _ExplodingLLM())

    rows, cypher, error = gr._run_cypher("小张养了什么")

    assert rows == []
    assert cypher == ""
    assert error.startswith("图谱不可用"), error
    # 若这里真的调了 LLM，_ExplodingLLM 会抛 AssertionError，用例失败。


# ─── 4. 最贴故障现场的回归：Neo4j 不可达时 import 也必须成功 ──────
def test_import_succeeds_when_neo4j_unreachable():
    """用不存在的端口做一次真实 import。

    这条是直接复现原故障的：只要模块顶层的连接又回来了，
    子进程就会以非 0 退出，用例立刻红。
    """
    import os
    import subprocess
    import sys

    env = dict(os.environ, NEO4J_URI="bolt://127.0.0.1:59999")
    proc = subprocess.run(
        [sys.executable, "-c", "import graph_retriever; print('IMPORT_OK')"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr[-800:]
    assert "IMPORT_OK" in proc.stdout
