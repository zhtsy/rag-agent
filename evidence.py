# -*- coding: utf-8 -*-
"""evidence.py —— 把检索工具的字符串输出，还原成结构化证据。

为什么需要这一层
----------------
检索工具 `search_knowledge_base` 的返回值是**给大模型读的字符串**：

    【来源：knowledge_base/人物志.md】【置信度：0.9312】
    小智住在南京，喜欢猫。

    ---

    【来源：知识图谱属性】【确定性事实】
    加里奥由禁魔石制成

这个格式是给 prompt 用的——模型直接看得懂出处和分数，不需要理解 JSON。

但一旦要把检索能力暴露给别的系统（MCP、HTTP API、评测脚本），字符串就不够了：
调用方需要知道「这是几条证据、各自多少分、是**根本没命中**、还是**检索出错**」。
这四件事在字符串里混在一起，人眼能分，机器不能。

本模块只做一件事：**忠实解析，不做任何判断**。
  - 不截断（top_k 由调用方决定，截断属于业务决策）
  - 不改写正文（改了就不是证据了）
  - 不把「没命中」和「出错」合并（这两种情况上层处理完全相反：
    没命中要如实告知用户，出错要重试或告警）

五种形态（样本全部来自真实输出，见 tests/test_evidence.py）
------------------------------------------------------------
1. evidence       向量/混合检索命中，带 `【来源】` 与 `【置信度】` 前缀
2. graph_evidence 图谱命中：`【图谱检索结果】【来源：a,b】` + 事实句 + 尾部置信度
3. graph_answer   只有事实句和尾部置信度、没有头部标签（旧形态 / 降级形态）
4. no_match       明确未命中（检索落空，属正常业务结果，不是故障）
5. error          检索内部异常（故障）
6. text           兜底：认不出结构时原样返回，绝不假装解析成功

为什么 2/3 和 1 必须分开：图谱的 `confidence` 是拿 reranker 给整段事实句打的分，
向量的 `confidence` 是给「单个文档块」打的分。两者数字都叫 confidence，
但一个句子的分和一个块的分不可比，合并统计会得出"平均置信度 0.8"这种无意义结论。
"""

import re
from typing import Any, Dict, List, Optional

# 与 rag_agent._search_knowledge_base_impl 的拼装方式严格对应。
# 改成 getattr 或更宽松的匹配都不行——一旦对不上，宁可走 text 兜底，
# 也不要"猜"出一个看似合理的结构。
SEPARATOR = "\n\n---\n\n"

# 块头：`【来源：x】【置信度：0.9312】` 或 `【来源：知识图谱属性】【确定性事实】`
_HEAD_RE = re.compile(
    r"^【来源：(?P<source>[^】]+)】"
    r"(?:【置信度：(?P<score>[0-9]+(?:\.[0-9]+)?)】|(?P<fact>【确定性事实】))?"
    r"\n?"
)

# 图谱事实句的置信度在末尾：`小智住在南京。【置信度：0.9312】`
_TAIL_SCORE_RE = re.compile(r"【置信度：([0-9]+(?:\.[0-9]+)?)】\s*$")

# 图谱命中的头部：`【图谱检索结果】【来源：a.txt, b.txt】\n`
# 来源可能是多个（逗号分隔），也可能是一个都没记到时的占位「本地知识图谱」，
# 所以用 `[^】]*` 而不是 `[^】]+`——空来源也要能匹配上。
_GRAPH_HEAD_RE = re.compile(r"^【图谱检索结果】【来源：(?P<sources>[^】]*)】[ \t]*\n?")

# 未命中：向量链路和图谱链路各有一句，都要认
_NO_MATCH_TEXTS = {
    "未在知识库中找到相关信息。",
    "图谱中未找到相关信息。",
}

# 故障前缀。顺序有意义：`图谱检索出错` 必须先于任何"未找到"判断，
# 否则会把"数据库连不上"读成"知识库没有这条资料"——那是把故障伪装成正常结果。
_ERROR_PREFIXES = (
    "检索工具执行失败",
    "检索问题为空",
    "图谱检索出错",
)


def _result(kind: str, raw: str, hits: Optional[List[Dict[str, Any]]] = None,
            message: str = "") -> Dict[str, Any]:
    return {
        "kind": kind,
        "hits": hits or [],
        "message": message,
        "raw": raw,
    }


def _parse_block(block: str) -> Optional[Dict[str, Any]]:
    """解析单个证据块。认不出结构返回 None（由调用方决定怎么兜底）。"""
    m = _HEAD_RE.match(block)
    if not m:
        return None
    score = m.group("score")
    return {
        "source": m.group("source"),
        "score": float(score) if score is not None else None,
        "content": block[m.end():].strip(),
        # graph_fact 是确定性事实（Cypher 直出），没有 rerank 分，
        # 所以 score 恒为 None。把它标出来，免得下游拿 None 去做平均。
        "kind": "graph_fact" if m.group("fact") else "vector",
    }


def _parse_graph(text: str) -> Optional[Dict[str, Any]]:
    """图谱命中：头部来源 + 多行事实 + 尾部置信度。

    置信度可能**不存在**——reranker 打分失败时 `_append_confidence` 直接返回原文，
    不补假分数。所以这里 score 允许是 None，不能拿 0 或 1 兜底。
    """
    m = _GRAPH_HEAD_RE.match(text)
    if not m:
        return None

    body = text[m.end():]
    score: Optional[float] = None
    tail = _TAIL_SCORE_RE.search(body)
    if tail:
        score = float(tail.group(1))
        body = _TAIL_SCORE_RE.sub("", body)

    names = [s.strip() for s in m.group("sources").split(",") if s.strip()]
    return {
        "source": " / ".join(names) if names else "本地知识图谱",
        "score": score,
        "content": body.strip(),
        "kind": "graph_evidence",
    }


def parse_retrieval(text: str) -> Dict[str, Any]:
    """把检索工具输出解析成 {kind, hits, message, raw}。

    `raw` 永远原样保留：调用方若觉得解析错了，还能拿到原文。
    """
    raw = text if isinstance(text, str) else ""
    t = raw.strip()

    if not t:
        return _result("error", raw, message="检索返回空内容")

    for prefix in _ERROR_PREFIXES:
        if t.startswith(prefix):
            return _result("error", raw, message=t)

    if t in _NO_MATCH_TEXTS:
        return _result("no_match", raw, message=t)

    # 图谱命中先判：它自带头部标签，且**不分块**（是整段事实，不是证据列表）。
    # 顺序不能和下面调换——图谱结果末尾也有置信度，先走分块会解析失败退到 text。
    graph_hit = _parse_graph(t)
    if graph_hit:
        return _result("graph_evidence", raw, hits=[graph_hit])

    hits: List[Dict[str, Any]] = []
    for block in t.split(SEPARATOR):
        if not block.strip():
            continue
        parsed = _parse_block(block)
        if parsed is None:
            # 只要有一块对不上，整体就不能当证据用。
            # 半解析的结果比不解析更危险——调用方会以为拿到了全部证据。
            hits = []
            break
        hits.append(parsed)

    if hits:
        return _result("evidence", raw, hits=hits)

    # 图谱路径：事实句 + 末尾置信度，没有来源标签
    m = _TAIL_SCORE_RE.search(t)
    if m:
        content = _TAIL_SCORE_RE.sub("", t).strip()
        return _result("graph_answer", raw, hits=[{
            "source": "知识图谱",
            "score": float(m.group(1)),
            "content": content,
            "kind": "graph_answer",
        }])

    # 兜底：认不出结构。原样交出去，并明确标注——不要伪装成解析成功。
    return _result("text", raw, hits=[{
        "source": "",
        "score": None,
        "content": t,
        "kind": "text",
    }])


def format_evidence(payload: Dict[str, Any], elapsed_ms: Optional[float] = None,
                    cached: Optional[bool] = None) -> str:
    """把解析结果渲染成模型可读的文本（MCP 工具返回用）。

    这里的措辞是**功能性**的，不是修饰：
    - no_match 要明确写出「不是故障、不要重试」，否则上层 agent 会换个说法反复查，
      把一次「知识库没有」放大成五六次无效检索；
    - error 要明确写出「可以重试」，与 no_match 给出相反的指令；
    - 每条证据都带出处和分数，模型才有依据判断该不该引用。
    """
    kind = payload.get("kind", "text")
    hits = payload.get("hits") or []

    meta = []
    if elapsed_ms is not None:
        meta.append(f"{elapsed_ms:.0f} ms")
    if cached is True:
        meta.append("命中缓存")
    elif cached is False:
        meta.append("未命中缓存")
    meta_str = f"（{'，'.join(meta)}）" if meta else ""

    if kind == "no_match":
        return (
            f"知识库中没有与这个问题相关的内容{meta_str}。\n\n"
            "这不是服务故障，而是知识库本身没有覆盖这个话题。"
            "请如实告知用户「知识库里没有这方面的资料」，"
            "不要换一种说法反复检索，也不要凭常识编造答案。"
        )

    if kind == "error":
        return (
            f"检索失败：{payload.get('message') or '未知错误'}\n\n"
            "这是服务端故障，不是知识库缺失，可以重试；"
            "若连续失败，请检查后端服务状态（/health）。"
        )

    if kind in ("graph_answer", "graph_evidence"):
        hit = hits[0] if hits else {"content": "", "score": None}
        score = hit.get("score")
        src = (hit.get("source") or "").strip()
        head = f"知识图谱回答{meta_str}："
        if kind == "graph_evidence" and src:
            head += f"\n来源：{src}"
        tail = f"\n（图谱事实可信度 {score:.4f}）" if isinstance(score, float) else ""
        return f"{head}\n{hit.get('content', '')}{tail}"

    if kind == "evidence":
        lines = [f"共检索到 {len(hits)} 条证据{meta_str}："]
        for i, hit in enumerate(hits, 1):
            src = hit.get("source") or "未知来源"
            if hit.get("score") is None:
                tag = "确定性事实"
            else:
                tag = f"相关度 {hit['score']:.4f}"
            lines.append(f"\n[{i}] 来源：{src} ｜ {tag}\n{hit.get('content', '')}")
        return "\n".join(lines)

    # text 兜底
    body = hits[0].get("content") if hits else payload.get("raw", "")
    return f"检索返回了未结构化的内容{meta_str}，原文如下：\n{body}"
