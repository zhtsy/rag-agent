# -*- coding: utf-8 -*-
"""接线契约测试：确认"该用公共模块的地方没有偷偷复制一份"。

为什么需要
──────────
这套系统摔得最狠的跤不是算法错，是**同一个逻辑存在两个实现**：
置信度有过两套算法（前端看到的数和护栏判断的数不一致），
拒答词表也有过两份。这类问题用行为测试测不出来——因为两份实现各自都"对"，
只有放在一起看才知道分叉了。

所以这里不执行 rag_agent（那要加载 1.2GB 模型），而是用 ast 读源码结构：
  · 置信度逻辑必须来自 confidence.py，rag_agent.py 里不许再定义一份
  · 请求计时必须来自 obs.py，main1.py 里必须真的调了 finish()
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _parse(name: str) -> ast.Module:
    return ast.parse((ROOT / name).read_text(encoding="utf-8"))


def _imported_names(tree: ast.Module, module: str) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names |= {alias.name for alias in node.names}
    return names


def test_rag_agent_uses_shared_confidence_module():
    tree = _parse("rag_agent.py")
    names = _imported_names(tree, "confidence")
    assert {"looks_like_abstention", "compute_confidence", "answer_status"} <= names


def test_rag_agent_does_not_redefine_confidence_logic():
    """一旦有人在这里重新写一份 `def _looks_like_abstention`，两套口径就开始漂了。"""
    tree = _parse("rag_agent.py")
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for forbidden in ("_looks_like_abstention", "_compute_confidence", "_answer_status"):
        assert forbidden not in defined, f"{forbidden} 又在本文件里定义了一份，请改成从 confidence 导入"


def test_rag_agent_instruments_retrieval_stages():
    """检索链路的关键阶段必须打点，否则"慢在哪"又只能靠猜。"""
    src = (ROOT / "rag_agent.py").read_text(encoding="utf-8")
    for stage in ('stage("rerank")', 'stage("retrieve")', 'stage("graph")', 'stage("decide")'):
        assert stage in src, f"缺少耗时打点 {stage}"


def test_main1_starts_and_finishes_trace():
    """main1 必须 start + finish 配对，否则埋点只进不出，日志里什么都看不到。"""
    src = (ROOT / "main1.py").read_text(encoding="utf-8")
    assert "_obs.start(" in src
    assert "_obs.finish(" in src
    assert '"timings"' in src          # 终帧里带上了分阶段耗时
