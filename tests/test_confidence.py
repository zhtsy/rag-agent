# -*- coding: utf-8 -*-
"""置信度 / 拒答判定的回归测试。

这些断言原来是手跑脚本（t_scope.py）里的 print + check，靠人记得去跑。
改成 pytest 之后至少解决三件事：失败会红、能进 CI、能一条条定位。

用例来源：全部是**线上真实踩过的坑**，不是编出来的边界。
每条都标了当初是什么问题，改了逻辑导致它红，先看注释再改断言。
"""

import pytest

from confidence import (
    CONFIDENCE_UNMEASURED,
    answer_status,
    compute_confidence,
    looks_like_abstention,
    split_sentences,
)


# ============================================================
# 1. 拒答判定
# ============================================================
@pytest.mark.parametrize("answer", [
    # 用户实际问的那条：正确拒答，但当时被判成"答得很差"（置信度 0）
    "知识库里没有关于马斯克的信息，无法从知识库中找到相关内容。",
    # 图谱链路的固定说法，不含"知识库中没有"，最容易漏
    "知识图谱中没有查询到相关信息。",
    "图谱中没有找到相关信息。",
    # 第一句就用弱词明说没有
    "资料中没有提到马斯克的职业。",
])
def test_abstention_detected(answer):
    assert looks_like_abstention(answer) is True


@pytest.mark.parametrize("answer", [
    # ★ 历史 bug：主问题答了、末尾补一句"某细节没写"，被当成拒答、置信度归零。
    #   弱词只在第一句生效，就是为了这四条。
    "小陈养了一只叫豆豆的金毛。资料没有说明它几岁了。",
    "小陈养了豆豆（金毛）。资料只提到这个名字，没有说明是什么品种",
    "小陈养了豆豆。资料中没有说品种。",
    "小陈养了豆豆。资料里没有写它的年龄。",
])
def test_answer_plus_caveat_is_not_abstention(answer):
    assert looks_like_abstention(answer) is False


def test_abstention_at_conclusion_position():
    """T031 真实答案：拒答落在最后一句（结论位），只扫第一句会漏判。

    当初的表现：这条被判成 answered、confidence=0.4457，
    于是"检索到了却没答出来"（missed）这种该排查的情况被掩盖了。
    """
    t031 = ("资料里没有直接写小陈住在哪个省，只提到她在广州的一家三甲医院工作，"
            "住在医院附近的一个小区，父母在老家、偶尔来广州看她。"
            "所以无法从知识库中确定她住在哪个省。")
    assert looks_like_abstention(t031) is True


def test_weak_word_at_tail_is_ignored():
    """弱词在结论位**不算**拒答——否则"…资料没有说明品种"会把正确答案判死。"""
    answer = "小陈养了豆豆。资料中没有说品种。"
    assert looks_like_abstention(answer) is False


@pytest.mark.parametrize("answer", ["", "   ", "\n\n"])
def test_empty_answer_counts_as_abstention(answer):
    """空答案没有承载任何事实，按拒答处理。"""
    assert looks_like_abstention(answer) is True


def test_abstention_ignores_lowercase_punctuation_only():
    """英文句点不参与断句：否则 "0.95" / "Mr. Smith" 会把句子切碎。"""
    assert split_sentences("得分 0.95。然后呢？") == ["得分 0.95", "然后呢"]


# ============================================================
# 2. 置信度取值
# ============================================================
def test_abstention_confidence_is_none():
    conf, sig = compute_confidence([0.0026, 0.0019],
                                  "知识库里没有关于马斯克的信息，无法从知识库中找到相关内容。")
    assert conf is None          # 不是 0.0：0 会被读成"答得很差"，而这条拒答是对的
    assert sig["reason"] == "abstention"
    assert sig["abstained"] is True


def test_confidence_is_max_not_mean():
    """取 max 而不是加权平均。

    当初用平均值把一次满分命中（0.9966）和一堆 0.01 的无关块混在一起，
    算出 0.4865，直接掉到所有阈值下面——这是纯数学错误。
    """
    conf, sig = compute_confidence([0.9966, 0.01, 0.01, 0.01], "小智喜欢喝咖啡。")
    assert conf == 0.9966
    assert sig["top_score"] == 0.9966
    assert sig["n_scores"] == 4


def test_no_scores_means_unmeasured_not_zero():
    """有证据但拿不到 rerank 分（图谱链路）→ 0.5「无度量」，不是 0，也不是 None。"""
    conf, sig = compute_confidence([], "小智喜欢喝咖啡。")
    assert conf == CONFIDENCE_UNMEASURED == 0.5
    assert sig["reason"] == "no_rerank_score"


def test_full_score_survives_rounding():
    """0.9966 必须原样出来。

    历史 bug：写进结果的是"加了关键词 boost 的排序分"（0.9966+0.05=1.0466），
    被下游 `<=1.0` 的过滤整条丢掉，于是满分题显示成 0.5（无度量）。
    """
    conf, _ = compute_confidence([0.9966], "小智喜欢喝咖啡。")
    assert conf == 0.9966
    assert conf <= 1.0


def test_confidence_is_rounded_to_4():
    conf, _ = compute_confidence([0.123456789], "随便答一句。")
    assert conf == 0.1235


def test_none_scores_are_ignored():
    """scores 里混进 None 不能炸（图谱链路偶尔给空值）。"""
    conf, sig = compute_confidence([None, 0.7], "小智喜欢喝咖啡。")
    assert conf == 0.7
    assert sig["n_scores"] == 1


# ============================================================
# 3. 四态交叉
# ============================================================
@pytest.mark.parametrize("abstained,has_evidence,expected", [
    (False, True, "answered"),      # 有证据 + 正常作答
    (True, False, "out_of_scope"),  # 无证据 + 拒答：知识库本来就没有，正常
    (True, True, "missed"),         # 有证据 + 拒答：检索到了却没答出来，要查
    (False, False, "ungrounded"),   # 无证据 + 硬答：幻觉风险，最该警惕
])
def test_answer_status_matrix(abstained, has_evidence, expected):
    assert answer_status(abstained, has_evidence) == expected


def test_user_reported_case_end_to_end():
    """把用户报的那条问题从头走一遍，确认修复没有回退。

    现象：问"根据知识库，马斯克是谁" → 答案正确（知识库没有），
    但置信度是 0（像答砸了）、来源还挂着 test2/test3。
    现在的预期：conf=None、status=out_of_scope、来源为空（来源那半在检索层修）。
    """
    answer = "知识库中没有关于马斯克的信息，无法从知识库中找到回答。"
    scores = [0.0026, 0.0019, 0.0019, 0.0015, 0.0010]   # 实测五块，全部约等于零
    has_evidence = False                                 # 来源被门槛清空
    conf, sig = compute_confidence(scores, answer)
    assert conf is None
    assert answer_status(sig["abstained"], has_evidence) == "out_of_scope"
