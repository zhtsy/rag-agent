# -*- coding: utf-8 -*-
"""规则快速路径的单元测试。

零第三方依赖，能进那个 0.2 秒的 CI。

最该保护的其实不是"规则能命中"，而是**规则不乱命中**：
一次误判要赔一次图谱往返（实测 1723ms）+ 一次降级，
比它省下的 1130ms 更贵。所以负例比正例重要。
"""

import pytest

from retrieval_router import (
    AGGREGATE_RULES,
    RULES,
    can_use_rules,
    decide,
    rule_aggregate,
    rule_relation,
)

# ----------------------------------------------------------------------
# 1. 正例：每条规则都必须有一条来自真实题库的问句
#    题库里没有的规则直接别加——加了就是没人验证过的规则
# ----------------------------------------------------------------------
POSITIVE = [
    ("residence", "根据知识库，小陈住在哪个省？"),
    ("residence", "小智是哪里人"),
    ("membership", "小陈属于哪个科室"),
    ("relation", "加里奥和德玛西亚是什么关系？"),
    ("relation", "小陈是谁的爸爸"),
    ("maker", "谁制作了加里奥？"),
    ("maker", "加里奥是由谁锻造的"),
    ("material", "加里奥是由什么材料制成的？"),
    ("material", "这座雕像用什么做的"),
    ("workplace", "小吴在哪里工作？"),
    ("workplace", "小陈就职于哪家医院"),
    ("pet", "小李养了什么宠物？"),
    ("pet", "小陈养的狗是什么品种？"),
    ("common", "小明和小王有什么共同点"),
]


@pytest.mark.parametrize("name,query", POSITIVE)
def test_each_rule_has_a_real_query(name, query):
    hit, matched = rule_relation(query)
    assert hit is True, f"{query!r} 应命中 {name}，实际未命中"
    assert matched == name, f"{query!r} 命中了 {matched}，期望 {name}"


def test_every_rule_is_covered_by_a_positive_case():
    """反向约束：加了规则却没测试，就无法证明它不会误判。"""
    covered = {name for name, _ in POSITIVE}
    declared = {name for name, _ in RULES}
    assert declared <= covered, f"这些规则没有正例：{sorted(declared - covered)}"


# ----------------------------------------------------------------------
# 2. 负例：属性查询、菜谱、玩法——这些是"像关系题但不是"
# ----------------------------------------------------------------------
NEGATIVE = [
    "小智喜欢什么？",              # 属性
    "川菜有什么特点",              # 属性
    "番茄炒蛋怎么做",              # 流程
    "泰拉瑞亚有什么玩法",          # 属性
    "根据知识库，小陈是做什么工作的？",  # 职业是属性，不是工作单位——真实误判回归
    "知识库里有哪些宠物相关的内容",      # 裸词"宠物"不算，必须带谓词
    "这几道菜哪个更简单",          # 比较
]


@pytest.mark.parametrize("query", NEGATIVE)
def test_attribute_queries_are_not_relation(query):
    hit, name = rule_relation(query)
    assert hit is False, f"{query!r} 被 {name} 误判为关系题"


def test_job_title_is_not_workplace():
    """C004 回归：离线比对 48 题时抓到的唯一误判，钉死它。

    为什么它值得一条独立测试：这条规则一旦回退，评测分数**不会变**
    （多跑一次图谱再降级，答案最终一样），只有耗时变差——
    这种"分数看不出来"的退化，只能靠单测拦。
    """
    assert rule_relation("小陈是做什么工作的")[0] is False
    assert rule_relation("小陈在哪里工作")[0] is True


# ----------------------------------------------------------------------
# 3. 启用条件：只有恰好一个实体时才可信
# ----------------------------------------------------------------------
@pytest.mark.parametrize("entities,expected", [
    ([], False),          # 无实体：谁养了猫这类聚合题该交给模型
    (["小智"], True),
    (["小智", "小明"], False),  # 多实体上游已分流
])
def test_can_use_rules(entities, expected):
    assert can_use_rules(entities) is expected


def test_no_entity_disables_rules():
    """约束 1 的守卫：把 can_use_rules 改成恒 True 会立刻变红。

    注意这里必须用**非聚合**的无实体问句。"谁养了猫"是聚合题，现在由
    AGGREGATE_RULES 接管（见下一组测试），不再走这条路径。
    """
    hit, reason, need_llm = decide("番茄炒蛋怎么做", [])
    assert (hit, reason, need_llm) == (False, "no_ent", True)


# ----------------------------------------------------------------------
# 3.5 聚合问句：天然没有主语实体，但必须走图谱
#
# C009 实测：「养猫的是哪几个人」被 LLM 改写成「养猫的人有哪些」，
# 规则未命中 + 实体 0 个 → 交给模型随机判，于是同一道题时通时不通。
# 这类问句的答案本身就是实体集合，不能要求它有主语实体。
# ----------------------------------------------------------------------
@pytest.mark.parametrize("query,expected", [
    ("谁养了仓鼠", "rule:agg_pet"),                    # C007 原话
    ("养猫的人有哪些", "rule:agg_pet_rev"),            # C009 改写后形态
    ("养猫的是哪几个人", "rule:agg_pet_rev"),          # C009 原话
    ("哪些人养了狗", "rule:agg_pet"),
])
def test_aggregate_questions_route_to_graph(query, expected):
    hit, reason, need_llm = decide(query, [])
    assert hit is True, f"{query!r} 应判为聚合题"
    assert reason == expected
    assert need_llm is False, "聚合题不该再发一次模型请求"


def test_aggregate_rules_do_not_catch_attribute_queries():
    """负例守卫：属性/流程问句不能被聚合规则吞进来。"""
    for q in ["番茄炒蛋怎么做", "川菜有什么特点", "小智喜欢什么"]:
        hit, _ = rule_aggregate(q)
        assert hit is False, f"{q!r} 被聚合规则误判"


# ----------------------------------------------------------------------
# 4. 决策输出的三种依据——它要能原样进日志
# ----------------------------------------------------------------------
def test_decide_rule_hit_skips_llm():
    hit, reason, need_llm = decide("小陈住在哪个省？", ["小陈"])
    assert hit is True
    assert reason == "rule:residence"
    assert need_llm is False, "规则命中就不该再发一次模型请求"


def test_decide_falls_back_to_llm():
    hit, reason, need_llm = decide("小智喜欢什么？", ["小智"])
    assert hit is False
    assert reason == "llm"
    assert need_llm is True, "规则判不了必须回退模型，不能自己猜"


def test_empty_query_is_safe():
    assert rule_relation("") == (False, "")
    assert rule_relation(None) == (False, "")
