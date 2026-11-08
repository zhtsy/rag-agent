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


def answer_status(abstained: bool, has_evidence: bool) -> str:
    """把「答没答」和「有没有证据」交叉成四种状态，供前端与排查使用。

      answered      有证据 + 正常作答        → 常规情况
      out_of_scope  无证据 + 拒答            → 知识库本来就没有，正常拒答
      missed        有证据 + 拒答            → 检索到了却没组织出答案，异常，要查
      ungrounded    无证据 + 硬答            → 没有依据还在答，幻觉风险，最该警惕
    """
    if abstained:
        return "missed" if has_evidence else "out_of_scope"
    return "answered" if has_evidence else "ungrounded"
