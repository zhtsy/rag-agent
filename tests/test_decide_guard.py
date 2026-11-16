# -*- coding: utf-8 -*-
"""工具决策兜底判据的单元测试。

零第三方依赖，能进那个 0.2 秒的 CI。

这里最该保护的**不是**"能兜底"，而是**不要乱兜底**：
每多兜一次就多跑一次检索（3–4 秒），对一个明确说了 need_tool=false 的寒暄
也强制检索，就是把延迟白白送给用户。

但反过来说，漏兜一次 = 静默答错，所以天平是向"多兜"这边偏的。
这批用例把那个偏斜的刻度钉死：
  · 只有严格 JSON 的 false 才算"模型说不查"
  · 自然语言不算（宁可多查）
  · 用户说"根据知识库"时，模型说 false 也不算（规则 2 优先级最高）
"""

import pytest

from decide_guard import (
    declares_need_tool,
    explicit_no_tool,
    has_substance,
    looks_legal,
    requires_retrieval,
    should_force_retrieval,
    strip_retrieval_prefix,
)

# ----------------------------------------------------------------------
# 1. requires_retrieval：只认指令式说法
# ----------------------------------------------------------------------

INSTRUCTION_POSITIVE = [
    "根据知识库，小智喜欢什么？",
    "根据知识库，谁住在广州？",
    "请根据资料回答：小陈养了什么宠物",
    "根据文档，川菜有什么特点",
    "查一下知识库，小周最拿手的菜",
    "从知识库告诉我小李的宠物",
    "按照知识库里写的，小刘多大了",
]


@pytest.mark.parametrize("q", INSTRUCTION_POSITIVE)
def test_instruction_phrases_detected(q):
    assert requires_retrieval(q) is True


@pytest.mark.parametrize(
    "q",
    [
        "你好",
        "",
        None,
        "知识库是什么？",          # 在问知识库本体，不是指令
        "知识库里大概有多少文档",   # 同样不是指令
        "小智喜欢什么？",           # 没有显式要求，交给模型判断
        "帮我写首诗",
    ],
)
def test_non_instruction_not_detected(q):
    assert requires_retrieval(q) is False


# ----------------------------------------------------------------------
# 2. explicit_no_tool：只认严格 JSON
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        '{"need_tool": false}',
        '  {"need_tool": false}  ',
        '```json\n{"need_tool": false}\n```',
        '```\n{"need_tool": false}\n```',
        '{"need_tool":false}',
        '好的，{"need_tool": false}',   # 前后夹了客套话也认，但只在 JSON 完整时
    ],
)
def test_explicit_false_recognized(content):
    assert explicit_no_tool(content) is True


@pytest.mark.parametrize(
    "content",
    [
        "",
        None,
        "   ",
        "不需要查询知识库",           # 自然语言否定：刻意不认
        "这个问题与知识库无关，直接回答即可。",
        '{"need_tool": true}',       # 说"要查"却没给 tool_call：属于沉默，必须兜底
        '{"need_tool": "false"}',    # 字符串不是布尔
        "query: 小智喜欢什么",       # 半截工具文本，没还原成调用
        "```json\n{\n```",           # 被截断的 JSON
    ],
)
def test_non_explicit_not_recognized(content):
    assert explicit_no_tool(content) is False


# ----------------------------------------------------------------------
# 3. should_force_retrieval：三条规则的优先级
# ----------------------------------------------------------------------


def test_silence_forces_retrieval():
    """空内容 = 沉默。这是 2026-11-07 那次的形态之一。"""
    assert should_force_retrieval("小智喜欢什么", "") is True
    assert should_force_retrieval("小智喜欢什么", None) is True
    assert should_force_retrieval("小智喜欢什么", "   ") is True


def test_garbage_forces_retrieval():
    assert should_force_retrieval("小智喜欢什么", "我看看……") is True


def test_explicit_false_with_plain_query_respected():
    """模型明确说不查、用户也没要求查 → 尊重，不加检索。"""
    assert should_force_retrieval("你好", '{"need_tool": false}') is False


def test_explicit_false_still_forces_when_user_asked():
    """规则 2：用户说了"根据知识库"，模型无权决定不查。

    这正是 T038 的可能形态——模型因为"广州不在已知实体里"而输出 false，
    但用户已经明确要求查库了。
    """
    assert should_force_retrieval("根据知识库，谁住在广州？", '{"need_tool": false}') is True
    assert should_force_retrieval("根据知识库，谁住在广州？", "") is True


def test_regression_t038_third_round():
    """回归用例：2026-11-07 第三轮 T038 的那次失败。

    形态未知（日志只记了"决策结果：不需要检索"），两种可能形态都必须兜住。
    """
    q = "根据知识库，谁住在广州？"
    for content in ("", None, "   ", '{"need_tool": false}', "这个问题知识库里没有"):
        assert should_force_retrieval(q, content) is True, f"未兜住: {content!r}"


# ----------------------------------------------------------------------
# 3b. 规则 5：法律问题不受模型 need_tool=false 影响
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        "商业银行的经营原则是什么？",
        "《民法典》第五百八十五条怎么规定的？",
        "专利法第三十六条",
        "劳动合同解除后经济补偿怎么算",
        "印花税借款合同的税率是多少",
        "违约金过高可以要求调整吗",
        "著作权侵权赔偿标准",
    ],
)
def test_legal_question_forces_retrieval_even_if_model_says_false(q):
    """2026-11-13 实测：qwen 把「商业银行的经营原则」判成不用查，
    于是不检索直接凭记忆答——`status=answered` 但 `sources=[]`，
    接口上完全看不出异常。规则 5 把"这类问题必须查"变成确定性逻辑。"""
    assert looks_legal(q, law_names=("商业银行", "民法典", "专利法")) is True
    assert should_force_retrieval(q, '{"need_tool": false}',
                                  law_names=("商业银行", "民法典", "专利法")) is True


def test_law_name_from_index_makes_question_legal():
    """最关键的一类：问句里没有书名号、没有条号、没有通用法律术语，
    但它点名了库里那部法（「商业银行」←《商业银行法》）。
    只有拿库里的真实法名去匹配才能抓到。"""
    assert looks_legal("商业银行的经营原则是什么？") is False        # 无名单时抓不到
    assert looks_legal("商业银行的经营原则是什么？", law_names=("商业银行",)) is True
    # 反过来：名单里没有这个词，就不该被误判
    assert looks_legal("怎么煮鸡蛋", law_names=("商业银行", "民法典")) is False


@pytest.mark.parametrize(
    "q",
    [
        "你好",
        "谢谢，再见",
        "这个方法可行吗",
        "我有个想法，你听听看",
        "没办法了",
        "今天几号",
        "帮我复述一下刚才的回答",
    ],
)
def test_chitchat_is_not_treated_as_legal(q):
    """**绝不能把裸的「法」字当信号**：方法/想法/没办法/说法都会误伤。
    误伤的后果是寒暄也去查库 → 检索落空 → 用户看到「未在知识库中找到相关信息」。"""
    assert looks_legal(q) is False
    assert should_force_retrieval(q, '{"need_tool": false}') is False


def test_legal_rule_does_not_break_multi_turn_answer_from_history():
    """规则 4（多轮里模型直接作答）不能被规则 5 顺手收走：
    闲聊只发生在非法律话题上；法律话题即使多轮也必须查库。"""
    assert should_force_retrieval("你好", "3", has_context=True) is False
    assert should_force_retrieval("民法典违约金规定", "3", has_context=True) is True



# ----------------------------------------------------------------------
# 4. strip_retrieval_prefix：伪造检索词时去掉指令前缀
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "q,expect",
    [
        ("根据知识库，小智喜欢什么？", "小智喜欢什么？"),
        ("根据知识库,小智喜欢什么？", "小智喜欢什么？"),
        ("根据资料：小陈养了什么宠物", "小陈养了什么宠物"),
        ("请根据知识库，谁住在广州？", "谁住在广州？"),
        ("按照知识库里写的，小刘多大了", "小刘多大了"),
        ("小智喜欢什么？", "小智喜欢什么？"),
    ],
)
def test_strip_prefix(q, expect):
    assert strip_retrieval_prefix(q) == expect


def test_strip_prefix_never_returns_empty():
    """纯前缀（没内容）不能 strip 成空串，否则检索词就没了。"""
    assert strip_retrieval_prefix("根据知识库") == "根据知识库"
    assert strip_retrieval_prefix("根据知识库，") == "根据知识库，"


# ----------------------------------------------------------------------
# 5. 规则 4：多轮上下文里的"抢答"不等于"沉默"（2026-11-12）
#
# 事故形态：18 轮会话后问"我最开始让你记住的编号是几？只回数字"，
# 决策模型直接输出 '3'（没给 tool_call），被判为沉默 → 强制检索 →
# 知识库里没有"我的编号" → 检索落空 → 回"无法从知识库中找到相关信息"，
# 而正确答案（1）就写在 49 字的摘要里。
# ----------------------------------------------------------------------


@pytest.mark.parametrize("content", ["3", "1", "ok", "是 1", "编号是 1"])
def test_context_answer_is_not_silence(content):
    """有历史时，模型吐出的实质内容视为"想直接答"，不强制检索。"""
    assert should_force_retrieval("我最开始让你记住的编号是几？", content, has_context=True) is False


@pytest.mark.parametrize("content", ["3", "ok", "", None, "   ", "……", "！？"])
def test_no_context_keeps_old_behavior(content):
    """单轮（评测的全部形态）行为必须与旧版逐字一致：非严格 JSON 一律兜底。

    这条是给 48 题基线兜底的护栏——放宽只在 has_context 为真时成立。
    旧逻辑是 `return not explicit_no_tool(content)`，所以这些输入全部 True。
    """
    assert should_force_retrieval("小智喜欢什么", content, has_context=False) is True


def test_context_does_not_rescue_empty_or_symbols():
    """有历史也不能把"纯符号/空白"当成回答——那还是沉默。"""
    for content in ("", None, "   ", "……", "！！！", "\n\n"):
        assert should_force_retrieval("我第一句问的是什么", content, has_context=True) is True


def test_context_still_yields_to_user_instruction():
    """规则 2 优先级最高：用户明说"根据知识库"，有历史也要查。"""
    assert should_force_retrieval(
        "根据知识库，谁住在广州？", "3", has_context=True) is True


def test_context_need_tool_true_is_protocol_failure():
    """说 need_tool: true 却没给 tool_call —— 协议失败，有历史也要兜底。"""
    assert should_force_retrieval(
        "我第一句问的是什么", '{"need_tool": true}', has_context=True) is True


def test_context_explicit_false_still_respected():
    """严格 JSON 的 false 本来就该放行，加不加 has_context 都一样。"""
    for flag in (True, False):
        assert should_force_retrieval("你好", '{"need_tool": false}', has_context=flag) is False


@pytest.mark.parametrize(
    "content,expect",
    [
        ("3", True),
        ("编号是 1", True),
        ("", False),
        ("   ", False),
        ("……", False),
        (None, False),
        (123, False),
    ],
)
def test_has_substance(content, expect):
    assert has_substance(content) is expect


@pytest.mark.parametrize(
    "content,expect",
    [
        ('{"need_tool": true}', True),
        (' {"need_tool":true} ', True),
        ('{"need_tool": false}', False),
        ("", False),
        (None, False),
    ],
)
def test_declares_need_tool(content, expect):
    assert declares_need_tool(content) is expect
