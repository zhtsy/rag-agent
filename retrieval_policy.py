# -*- coding: utf-8 -*-
"""检索的「选块策略」——纯函数，零依赖。

为什么要把这几行从 `rag_agent.py` 里搬出来
──────────────────────────────────────────
这个文件里没有任何模型、没有 langchain、没有 jieba。原因是**可测试性**：
`rag_agent.py` 一 import 就会加载 chroma + bge-reranker（约 15 秒 / 1.2GB），
真正出过 bug 的那几行逻辑因此从来没被自动化测试覆盖过 —— 每次改完只能靠
跑 38 条 eval 看总分，而总分是模型生成的，抖动足以把 bug 掩盖掉。

而这里编码的三条规则，每一条都对应一次线上事故：

  1. `DEFAULT_RERANK_CANDIDATES = 10`
     曾是 6。T019「加里奥为什么能吸收魔法」的 12 个候选里，含「禁魔石」的
     3 块在 RRF 里排下标 6/7/8，被截在门外，答案永远吐不出「禁魔石」。

  2. 排序分（含关键词加成）与原始分**分开存放**
     曾把加成后的 0.9966+0.05=1.0466 当成置信度写进结果，超过 1 后被下游
     `<= 1.0` 的过滤整条丢掉，满分题于是显示成 0.5（"无度量"）。

  3. 取前 5 必须**纯按分数**
     曾用「关键词命中硬占 4 个名额 + 另一队列补 1 个」，导致第 5 高分
     （0.4942，含禁魔石）被砍，换进一个 0.285 的低分块 —— 高分被低分挤掉。

测试在 `tests/test_retrieval_policy.py`，其中 `test_window_6_drops_forbidden_stone`
就是"把窗口改回 6 会立刻变红"的那一条。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

# ========== 策略常量（唯一的定义点，rag_agent 从这里取） ==========

# rerank 窗口。别再往下调，理由见文件头第 1 条；上调只增加耗时（实测 8 与 10 的
# top-5 完全一致，取 10 是给别的查询留余量）。
DEFAULT_RERANK_CANDIDATES = 10

# 相关性门槛。rerank 分回答的是"这块与问题有多相关"，而向量检索 + RRF 无论相关
# 与否都会返回 top-k。实测越界问题（"马斯克是谁"）top1 只有 0.0026~0.0535，
# 正常问题最低 0.6078，中间差一个数量级，0.15 落在空档里。
DEFAULT_MIN_RELEVANT = 0.15

# 关键词加成的权重。**默认关闭（0.0）**，机制保留，见下。
#
# 为什么从 0.05 改成 0
# ────────────────────
# 它原本的意图是"只在分数接近时做倾向"，但量级从来没对上过：
#
#   · 加成值 0.05
#   · 实测 rerank 分差 0.0019 ~ 0.005（同一题相邻名次之间）
#   · 0.05 是分差的 10~26 倍
#
# 所以它不是"微调"，是硬性优先。拿真实候选做对照（2026-11-24，5 道法律题，
# 同一批候选、同一批重排分，只改 boost）：4 道题的 top-5 完全一致，1 道题的
# 第 1 名与第 8 名互换。看着"没影响"，是因为法条块几乎**全部**命中关键词
# （实测 11/11、13/13、18/19、17/19）——大家一起加 0.05 等于都没加，
# 加成退化成常数。"没问题"是巧合，不是设计成立。
#
# 真正的风险场景恰好是向量检索最擅长的场景：**正确答案的措辞不含问题原词**。
# 问「被辞退能拿多少补偿」，条文写的是「经济补偿」「违法解除劳动合同」；
# 那 0.05 足够把它挤到 top-5 之外。而"同义改写"正是这条链路要解决的问题，
# 用它去惩罚同义改写，方向是反的。
#
# 保留机制（`select(..., boost=...)` 仍然生效）是为了两件事：
#   1. tests 里那两条事故测试仍能验证"加成不污染 raw"；
#   2. 将来真找到量级合适的值（< 0.002），改这一行就能开。
KEYWORD_BOOST = 0.0

# 最终喂给大模型的内容块数量。
TOP_K = 5

# 参与关键词加成的词长度下限、以及不承载检索意图的词。
MIN_KEYWORD_LEN = 2
STOPWORDS = frozenset({
    "什么", "为什么", "怎么", "如何", "哪里", "哪个", "介绍", "告诉",
    "根据", "知识库", "的", "了", "是", "能", "会", "被", "把", "和", "与",
})


@dataclass
class Selection:
    """选块结果。

    items 里每项是 `(原始下标, 排序分, 原始 rerank 分)`：
      · 排序分（含关键词加成）只用来**排名次**；
      · 原始分才是"这块与问题有多相关"的真实度量，置信度必须取它。
    两者分开存就是为了不再犯文件头第 2 条那个错。
    """

    items: List[Tuple[int, float, float]]
    best_raw: float
    relevant: bool
    n_candidates: int

    @property
    def indices(self) -> List[int]:
        return [i for i, _, _ in self.items]

    @property
    def raw_scores(self) -> List[float]:
        return [r for _, _, r in self.items]


def fallback_tokenize(text: str) -> List[str]:
    """不依赖 jieba 的粗切：按非中文/非字母数字边界切开。

    在线链路用的是 jieba（词边界准得多），这里只给"没装 jieba 的环境"兜底，
    比如 CI。测试用它可以不引入 jieba 依赖。
    """
    import re

    return re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z0-9]+", text or "")


def extract_keywords(
    query: str, tokenizer: Callable[[str], Sequence[str]] = None
) -> List[str]:
    """从 query 里取用于加成的关键词：长度 ≥2 且不在停用词表里。"""
    tok = tokenizer or fallback_tokenize
    return [
        w for w in tok(query or "")
        if len(w) >= MIN_KEYWORD_LEN and w not in STOPWORDS
    ]


def candidate_slice(total: int, max_candidates: int) -> range:
    """rerank 之前要打分的那一段下标。

    截断必须发生在 **rerank 之前**（打分耗时正比于候选对数），所以这里返回
    下标区间而不是过滤后的列表。`max_candidates <= 0` 视为不截断。
    """
    if max_candidates is None or max_candidates <= 0:
        return range(total)
    return range(min(total, max_candidates))


def candidate_window(index_size: int, default: int = DEFAULT_RERANK_CANDIDATES) -> int:
    """按索引规模决定 rerank 候选窗口（纯函数，可单测）。

    为什么不能用一个常量跑所有库
    ────────────────────────
    145 块时 10 够用（上面 T019 的教训是 6 不够）。但语料涨到 3000 块后，
    正确块经常落在第 11–20 名：实测 v3 语料上「苏州瑞运物流的对接人是谁」，
    20 个候选里正确那块排在第 12 位，被截在门外 → top1 只有 0.1096 →
    低于门槛 → 判"检索落空"→ 拒答。**问题不在检索没找到，在窗口太小。**

    代价是线性增加的重排耗时（每对约 0.3 s CPU），所以分档而不是一律放大：
    生产规模（≤300 块）保持原值，行为完全不变。
    """
    if index_size <= 0:
        return default
    if index_size <= 300:
        return default
    if index_size <= 1000:
        return max(default, 16)
    return max(default, 24)


def rerank_window(index_size: int, override=None,
                  default: int = DEFAULT_RERANK_CANDIDATES) -> int:
    """实际生效的窗口：环境变量只能**调大**，压不过规模下限。

    `.env` 里写着 `RAG_RERANK_CANDIDATES=10`，而代码原来写的是
    `env or candidate_window(...)` —— 环境变量永远为真，自适应窗口一次都没生效过。
    切到 7,000 块语料后日志仍是 `rerank 候选截断: 20 → 10`：正确块落在第 11–20 名
    时照样被截掉，"按规模放大窗口"这个修复等于白加。

    所以这里把两者的关系定死：规模下限是**地板**，环境变量只能在上面加，
    不能把它按回地板以下。要极端省算力时把 override 调大、把语料裁小，而不是
    悄悄把窗口压窄——那种省法只会换来"检索命中却答不出来"。
    """
    base = candidate_window(index_size, default)
    if override is None or (isinstance(override, str) and not override.strip()):
        return base
    try:
        want = int(override)
    except (TypeError, ValueError):
        return base
    if want <= 0:                      # 0/负数沿用"不截断"的旧语义 → 交回默认档
        return base
    return max(base, want)


def rerank_budget(n_candidates: int, override=None) -> int:
    """实际送去重排的块数：**默认全量**，环境变量只能调大。

    为什么不按索引规模分档了
    ────────────────────────
    `candidate_window(index_size)` 的判据是**索引块数**，但候选数**由 k 决定，
    与索引规模无关**：31,856 块的库，一次查询仍然只有 BM25 的 10 + 向量的 10。

    实测（2026-11-24，5 道法律题）候选数 11 / 13 / 19 / 20 / 19，
    全部小于 24 —— 那条"300 以上用 16、1000 以上用 24"的分档规则，
    从上线起**一次都没触发过**。日志里那句 `rerank 候选截断: 20 -> 10`
    是 k=30 时代的残留，现在连这条日志都不会再打出来。

    所以真正的约束一直是 k=10：正确块如果在 BM25 里排第 15、向量里排第 18，
    它根本进不了候选池，窗口设成 240 也捞不回来。**窗口和 k 是两个管同一件事的
    旋钮，其中一个是空的** —— 这就是设计缺陷的形态：不是参数调得不对，
    是这个参数不该存在。

    现在的语义：窗口 = 候选数（永不截断）。要给重排省算力就去改 k，
    那才是真正决定候选池大小的变量。`override` 只在**比候选数更大**时生效
    （保留旧语义，避免 .env 里一个过时的 10 把窗口悄悄压窄）。

    旧函数 `candidate_window` / `rerank_window` 保留不删：tests 里有 6 条测试
    钉着它们（包括"窗口改回 6 会立刻变红"那条），它们记录的是历史事故，
    删掉等于把教训一起删了。
    """
    n = max(0, int(n_candidates or 0))
    if override is None or (isinstance(override, str) and not override.strip()):
        return n
    try:
        want = int(override)
    except (TypeError, ValueError):
        return n
    if want <= 0:                      # 0/负数 = 没配，用默认（全量）
        return n
    return max(n, want)


def select(
    contents: Sequence[str],
    scores,
    keywords: Sequence[str] = (),
    *,
    min_relevant: float = DEFAULT_MIN_RELEVANT,
    top_k: int = TOP_K,
    boost: float = KEYWORD_BOOST,
) -> Selection:
    """按 rerank 分选块，并判断这一轮检索是否"落空"。

    contents 与 scores 必须一一对应（scores 是这些块的 rerank 原始分）。
    `scores` 允许是标量：只有一个候选时 `compute_score` 返回 float 而不是列表，
    不归一化的话下游 zip 会直接抛错（这是踩过的坑之一）。
    """
    if isinstance(scores, (int, float)):
        scores = [float(scores)]
    scores = [float(s) for s in scores or []]

    if len(contents) != len(scores):
        raise ValueError(
            f"contents 与 scores 数量不一致：{len(contents)} vs {len(scores)}"
        )
    if not contents:
        return Selection(items=[], best_raw=0.0, relevant=False, n_candidates=0)

    ranked: List[Tuple[int, float, float]] = []
    for idx, (content, raw) in enumerate(zip(contents, scores)):
        # 加成只进"排序分"，raw 原样保留
        sort_score = raw + (boost if any(kw in content for kw in keywords) else 0.0)
        ranked.append((idx, sort_score, raw))

    # 纯按排序分降序；同分时按下标升序，保证结果稳定可复现
    ranked.sort(key=lambda x: (-x[1], x[0]))

    best_raw = max(scores)
    relevant = best_raw >= min_relevant
    return Selection(
        items=ranked[:top_k] if relevant else [],
        best_raw=best_raw,
        relevant=relevant,
        n_candidates=len(contents),
    )


def keyword_hit_count(
    contents: Sequence[str], indices: Sequence[int], keywords: Sequence[str]
) -> int:
    """选中的块里有几块命中了关键词——只用于日志，不参与决策。"""
    return sum(
        1 for i in indices
        if i < len(contents) and any(kw in contents[i] for kw in keywords)
    )


def describe(sel: Selection) -> Dict[str, object]:
    """给日志/埋点用的摘要（不含正文，避免日志满天飞）。"""
    return {
        "n_candidates": sel.n_candidates,
        "best_raw": round(sel.best_raw, 4),
        "relevant": sel.relevant,
        "picked": len(sel.items),
        "raw_scores": [round(r, 4) for r in sel.raw_scores],
    }
