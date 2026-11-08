# -*- coding: utf-8 -*-
"""answer_guard 的回归测试（答案完整性兜底）。

它只做一件事：模型答了名字却丢了类别时，把检索原文里的那句补在末尾。
风险全在**误判**上——补错一句，用户会以为系统在乱加东西。
所以这里的反例比正例更重要：宁可漏补，不可错补。
"""

import pytest

import answer_guard as g

pytestmark = pytest.mark.skipif(
    not g.enabled(), reason="RAG_ANSWER_GUARD=0，兜底已整体关闭"
)

CTX_ZHANG = [
    "我不太喜欢运动…我住在上海静安区的一个老小区里…"
    "阳台上养了几盆多肉，还有一只叫“布丁”的仓鼠。"
]


# ============================================================
# 1. 该补的要补
# ============================================================
def test_supplements_missing_category():
    """"小张养了布丁" 丢了"仓鼠" —— 这正是当初用户报的问题。"""
    out = g.completeness_supplement("根据知识库，小张养了布丁。", CTX_ZHANG)
    assert "仓鼠" in out
    assert "布丁" in out


def test_supplements_graph_style_parens():
    """图谱格式「X（Y）」也要认。"""
    ctx = ["【图谱检索结果】【来源：test6.txt】\n小张养了布丁（仓鼠）。【置信度：0.9975】"]
    out = g.completeness_supplement("根据知识库，小张养了布丁。", ctx)
    assert "仓鼠" in out


# ============================================================
# 2. 不该补的一个字都不能加
# ============================================================
@pytest.mark.parametrize("answer,ctxs", [
    # 已经答全了
    ("根据知识库，小张养了一只叫布丁的仓鼠。", CTX_ZHANG),
    ("小张养了布丁，是只仓鼠。", ["小张养了布丁（仓鼠）。"]),
    # 问的根本不是这件事
    ("根据知识库，小张是一名平面设计师。", CTX_ZHANG),
    # 没有上下文可依据
    ("根据知识库，小张养了布丁。", []),
    ("根据知识库，小张养了布丁。", None),
    # 别的实体的属性，不能挪过来用
    ("小陈养了豆豆。", ["小张养了布丁（仓鼠）。"]),
])
def test_does_not_touch_when_not_needed(answer, ctxs):
    assert g.completeness_supplement(answer, ctxs) == ""


def test_enumeration_is_left_alone():
    """属性对超过 3 个时视为"枚举列表"，一律不补。

    否则会把一堆没问到的实体刷到答案尾巴上（"小智是…；小陈是…；小吴是…"），
    用户看着就像系统在自言自语。
    """
    ctx = ["小张（平面设计师），小李（设计师），小陈（医生），小吴（运营）"]
    assert g.completeness_supplement("小张是设计师。", ctx) == ""


@pytest.mark.parametrize("kind", ["那个人", "那只", "时候", "东西"])
def test_demonstratives_are_not_treated_as_kind(kind):
    """指示代词/泛化词不能被当成"类别"补进答案。

    首版就踩了：把 "叫小明的那个人" 里的"那个人"当类型，补出
    "（补充：小明是那个人。）"——纯噪声。
    """
    assert g._is_usable_kind(kind) is False


@pytest.mark.parametrize("kind", ["仓鼠", "橘猫", "平面设计师", "三甲医院"])
def test_real_kinds_are_usable(kind):
    assert g._is_usable_kind(kind) is True


def test_supplement_is_deterministic():
    """同样的输入必须给同样的输出：它是护栏，不能自己带随机性。"""
    first = g.completeness_supplement("根据知识库，小张养了布丁。", CTX_ZHANG)
    second = g.completeness_supplement("根据知识库，小张养了布丁。", CTX_ZHANG)
    assert first == second


def test_at_most_two_items():
    """最多补 2 条：补得越多，越像系统在抢话。"""
    ctx = ["有一只叫“布丁”的仓鼠。", "是一只橘猫，名字叫“代码”的那只。",
           "还有一只叫“拿铁”的猫。"]
    out = g.completeness_supplement("小张养了布丁和代码和拿铁。", ctx)
    assert out.count("补充") <= 2
