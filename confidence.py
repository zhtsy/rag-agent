# -*- coding: utf-8 -*-
"""confidence.py —— 置信度与拒答判定的唯一入口。

为什么单独一个文件
──────────────────
这套逻辑原来埋在 rag_agent.py 第 1500 行附近。想在 CI 里测它，就得先 import
rag_agent —— 而那会加载 chroma + bge + Neo4j 驱动，单次 import 十几秒、
1.2GB 内存。结果是：没人测，只能靠"改完手工跑一遍线上"，回归全靠人肉。
搬到这里之后它只依赖 re，`pytest tests/` 可以在一秒内跑完，进 CI 没有任何负担。

设计约束（这是之前出问题的根源，写在这里避免以后再分叉）
────────────────────────────────────────────────────
1. 置信度只能有一个生产者。原来同步/评估路径和线上 SSE 路径各算一套，
   前端看到的数和护栏判断的数不是同一个，于是出现"答对的题 0.09、拒答的题 0.81"。
2. 只能用检索证据算，不能用常数。历史上的 0.9500 / 0.8500 / 0.8 都是造出来的数。
3. 统计量取 max 而不是加权平均。rerank 分是逐块的相关性，一条回答的可信度由
   "最好的那块证据"决定；把它和一堆 0.01 的无关块做加权平均，好命中会被稀释成
   0.4865（实测），直接掉到任何阈值以下——这是纯数学错误，不是参数没调好。
4. 拒答不是"低置信回答"，也不该记成 0 分。系统已经明说"知识库没有"，
   这份答案不承载任何事实，记 0 会让人读成"答得很差"，记 1 又会读成"满分答案"。
   所以拒答的 confidence 是 None（不适用），好/坏由 status 区分。
"""

import re
from typing import Sequence, Tuple, Optional, Dict, Any

# 拒答措辞分两级，因为它们的"证据强度"不一样。
#
# 强词：这句话是在说**整条回答没有答案**（"无法从知识库中确定…"）。
#       它既可能出现在第一句，也常出现在结论位（"所以无法从知识库中确定她住在哪个省"），
#       所以除第一句外再看最后一句。实测 T031 就是这种："资料里没有直接写…
#       所以无法从知识库中确定她住在哪个省。" —— 只扫第一句会漏判，
#       于是这条拒答被算成 0.4457 的"正常回答"。
STRONG_ABSTAIN = (
    "无法从知识库", "无法在知识库", "未在知识库", "知识库中没有", "知识库中未",
    "知识库里没有", "无法从资料",
    # 图谱链路的固定说法（"知识图谱中没有查询到相关信息"）不含"知识库中没有"，
    # 漏了它，图谱拒答会被当成正经答案、拿到 0.5 的"无度量"分。
    "图谱中没有", "没有查询到",
)

# 弱词：也可能只是"某个细节没说明"，**只在第一句生效**。
# 这是历史踩过的坑："根据知识库里的资料，小陈养了豆豆。资料只提到这个名字，
# 没有说明是什么品种" —— 前半句是正经答案、后半句是补充声明，
# 原来按前 40 字扫，把这条正确答案判成了拒答，置信度直接归零。
# 注意 "资料中没有/资料里没有" 属于弱词：它们既能表示"整条没有"，也能表示
# "某个细节没写"（"资料中没有说品种"），放强词表会误杀后一种。
WEAK_ABSTAIN = (
    "无法回答", "无法确定", "没有提到", "未提及", "没有说明",
    "找不到", "未找到", "没有找到",
    "资料中没有", "资料里没有", "没有直接写",
)

# 有证据但拿不到 rerank 分（图谱打分失败等）时的中性值：语义是「无度量」，不是「中等把握」
CONFIDENCE_UNMEASURED = 0.5

# 断句：中文句号/问号/感叹号/换行。注意英文句点不参与——
# 答案里常出现 "0.95"、"Mr. Smith" 这类，按 "." 断会把句子切碎。
_SENTENCE_SPLIT = re.compile(r"[。！？!?\n]")


# 「自认无据 + 仍给数值」——幻觉补偿结构（2026-11-15 加）。
#
# 实例（reports/ablation/C.json 的 MX01，表格块被剔除后）：
#   「…但是，资料中并未直接给出借款合同的具体印花税率。通常情况下，借款合同的
#     印花税率是万分之零点五（0.0005）。」
# 前半句承认证据里没有，后半句还是给了数。外部看是满分答案（得分 1.00、
# confidence 0.9524），其实数值纯属模型记忆——恰恰猜对了，于是没人发现。
#
# 为什么不能简单加进 WEAK_ABSTAIN：同一批里 MX04 也在说「具体税率并未直接给出」，
# 但它后面只说"由省级政府决定"，没有编数，那是**诚实回答**，不该判成幻觉。
# 所以判据必须带上「声明之后确实出现了具体数值」这个条件。
_NO_EVIDENCE_PHRASES = (
    "并未直接给出", "没有直接给出", "未直接给出",
    "并未明确给出", "没有明确给出", "未明确给出",
    "并未给出具体", "没有给出具体", "未给出具体",
    "未提及具体", "没有提及具体",
)

# 「具体数值」= 带单位的阿拉伯数字，或「万分之零点五」这类中文分数表达。
# 单位只收金额/面积/体积/倍数/百分比：这些是"编出来的数"最典型的形态。
# 刻意不收「年/天/个月」——年份（"2021 年"）会大量误伤，而刑期多是
# 「三年以下」这种中文数字，本来也不在这个正则的覆盖范围里。宁可漏抓，不要误抓：
# 误抓的代价是把诚实回答标成幻觉，比漏掉一条更难解释。
_QTY_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:%|‰|％|万元|亿元|元|吨|平方米|立方米|倍)"
    r"|[零一二三四五六七八九十百千万]+分之[零一二三四五六七八九十]+"
)


def split_sentences(text: str) -> list:
    """把答案切成句子列表（去掉空串）。空输入返回空列表。"""
    if not text:
        return []
    return [s for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def looks_like_abstention(answer: str) -> bool:
    """判断这条回答是不是「知识库里没有」。

    判定顺序（越靠前越可信）：
      1. 第一句命中强词 / 弱词   → 拒答
      2. **最后一句**命中强词     → 拒答（结论位，如"所以无法从知识库中确定…"）
      3. 其余                     → 不是拒答

    为什么最后一句只认强词：弱词出现在末尾太常见了（"…资料没有说明品种"
    这种补充声明挂在结尾），认了就会把正确答案判成拒答。
    """
    sentences = split_sentences((answer or "").strip())
    if not sentences:
        # 空答案按拒答处理：没有内容就没有可承载的事实
        return True

    first, last = sentences[0], sentences[-1]
    if any(m in first for m in STRONG_ABSTAIN):
        return True
    if any(m in first for m in WEAK_ABSTAIN):
        return True
    if any(m in last for m in STRONG_ABSTAIN):
        return True
    return False


def hedged_guess(answer: str) -> bool:
    """答案里有没有「先承认没证据、紧接着给出具体数值」的结构。

    只看**声明所在句（声明之后的部分）和下一句**。再往后就进入别的话题了，
    那里出现金额很正常（"第四十八条规定罚款五万元"），扫全文会把诚实回答误判。

    与 looks_like_abstention 的关系：那个函数判"整条回答没答案"，这个函数判
    "有答案但数值没依据"。两者可以同时成立，也可以都不成立——它们是正交的两件事。
    """
    sentences = split_sentences((answer or "").strip())
    for i, sent in enumerate(sentences):
        for phrase in _NO_EVIDENCE_PHRASES:
            pos = sent.find(phrase)
            if pos < 0:
                continue
            # 同一句里声明之后就有数（"资料未给出具体税率，一般约 3%"）
            if _QTY_RE.search(sent[pos + len(phrase):]):
                return True
            # 下一句才给数（"…并未直接给出。通常情况下是万分之零点五"）
            if i + 1 < len(sentences) and _QTY_RE.search(sentences[i + 1]):
                return True
    return False


def compute_confidence(scores: Sequence[float],
                       answer: str) -> Tuple[Optional[float], Dict[str, Any]]:
    """唯一的置信度计算入口。返回 (confidence, signals)。

    confidence 语义：**对这条回答内容的把握**，不是"系统答对了的概率"。
    取值：
      · 拒答                    → None（不适用，见下）
      · 有 rerank 分            → 最高分（最好那块证据的相关性）
      · 没有任何 rerank 分      → CONFIDENCE_UNMEASURED（无度量）

    「拒答为什么是 None 而不是 0.0」——这是用户实际反馈出来的问题：
    问"根据知识库，马斯克是谁"，系统正确回答"知识库里没有马斯克的信息"，
    前端却显示置信度 0，看起来像"系统答得很差"。但这条回答并没有答错，
    它只是**没有可承载事实的答案**，因此 confidence 这个量对它不适用。
    0 和 1 都会被误读（0 = 答砸了，1 = 满分答案），所以这里给 None，
    由 status 字段区分（见 answer_status）：
      · out_of_scope  检索确实没命中 → 正常拒答，预期内
      · missed        检索命中了却答不出 → 真问题，需要排查
    """
    score_list = [s for s in (scores or []) if s is not None]
    top = max(score_list) if score_list else None
    abstained = looks_like_abstention(answer)

    if abstained:
        conf = None
        reason = "abstention"
    elif top is None:
        conf = CONFIDENCE_UNMEASURED
        reason = "no_rerank_score"
    else:
        conf = round(float(top), 4)
        reason = "rerank_top1"

    return conf, {
        "top_score": top,
        "n_scores": len(score_list),
        "abstained": abstained,
        "reason": reason,
    }


def answer_status(abstained: bool, has_evidence: bool, hedged: bool = False) -> str:
    """把「答没答」「有没有证据」「数值有没有依据」交叉成五种状态。

      answered      有证据 + 正常作答        → 常规情况
      out_of_scope  无证据 + 拒答            → 知识库本来就没有，正常拒答
      missed        有证据 + 拒答            → 检索到了却没组织出答案，异常，要查
      ungrounded    无证据 + 硬答            → 没有依据还在答，幻觉风险，最该警惕
      hedged        自认无据却给了具体数值   → 幻觉补偿，**最隐蔽的一种**：
                                              答案读起来有据（"资料中并未直接给出，
                                              但通常情况下是…"），数值却是模型自己
                                              补的，猜对了就完全看不出来。

    hedged 优先于 answered/ungrounded：它描述的是"这份答案里有一类内容没有任何
    证据支撑"，这个事实比"有没有检索到证据"更该被看见。
    """
    if abstained:
        return "missed" if has_evidence else "out_of_scope"
    if hedged:
        return "hedged"
    return "answered" if has_evidence else "ungrounded"
