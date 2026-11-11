# -*- coding: utf-8 -*-
"""memory_context 纯函数测试（不连库、不加载模型）。

这些函数是记忆链路里最容易出错的部分：
  · 截断切在半个句子里，下一轮模型就"忘了"关键限定
  · 长期记忆渲染漏了"只作背景"的护栏，模型会拿用户画像当证据
  · 模型吐 Markdown 围栏 / 单对象 / 英文字段，解析直接崩
"""

import pytest

from memory_context import (
    build_extract_prompt,
    evidence_support,
    parse_facts,
    render_long_term_block,
    resolve_session_id,
    smart_truncate,
)


# =====================================================================
# 1. 截断
# =====================================================================
def test_short_text_kept_as_is():
    assert smart_truncate("布丁是一只仓鼠。", 50) == "布丁是一只仓鼠。"


def test_truncate_prefers_sentence_end():
    text = "代码是一只橘猫，它的主人是小李。2024 年 3 月搬到深圳，住在南山区。"
    out = smart_truncate(text, 20)
    # 20 字以内最后一个句号落在"小李。"，必须断在这里，而不是切在"3 月搬"
    assert out.endswith("。")
    assert "小李" in out
    assert len(out) <= 22


def test_truncate_falls_back_to_comma_with_ellipsis():
    text = "这是一段没有任何句号的长句子，只有逗号在中间，后面还有很多很多内容"
    out = smart_truncate(text, 20)
    assert out.endswith("…")


def test_truncate_hard_cut_marks_ellipsis():
    text = "abcdefghijklmnopqrstuvwxyz"
    out = smart_truncate(text, 10)
    assert out == "abcdefghij…"


def test_truncate_edge_cases():
    assert smart_truncate("", 10) == ""
    assert smart_truncate(None, 10) == ""
    assert smart_truncate("abc", 0) == ""


# =====================================================================
# 2. 会话 ID 兜底（修 default_session 串台）
# =====================================================================
def test_real_session_id_passthrough():
    assert resolve_session_id("abc-123") == "abc-123"


@pytest.mark.parametrize("placeholder", ["default_session", "default", "", "   "])
def test_placeholder_gets_anonymous_id(placeholder):
    sid = resolve_session_id(placeholder, client_host="10.0.0.1", user_agent="UA")
    assert sid.startswith("anon-")
    assert sid != "default_session"


def test_same_client_same_anonymous_session():
    a = resolve_session_id("", client_host="10.0.0.1", user_agent="UA")
    b = resolve_session_id("default_session", client_host="10.0.0.1", user_agent="UA")
    assert a == b          # 同一设备刷新页面不应换会话


def test_different_client_different_session():
    a = resolve_session_id("", client_host="10.0.0.1", user_agent="UA")
    b = resolve_session_id("", client_host="10.0.0.2", user_agent="UA")
    assert a != b


def test_no_fingerprint_gives_unique_session():
    a = resolve_session_id("")
    b = resolve_session_id("")
    assert a != b and a.startswith("anon-")


# =====================================================================
# 3. 长期记忆渲染
# =====================================================================
def test_render_empty():
    assert render_long_term_block([]) == ""
    assert render_long_term_block([{"key": "", "value": "x"}]) == ""


def test_render_contains_guard_sentence():
    block = render_long_term_block([{"key": "称呼", "value": "小王"}])
    assert "称呼：小王" in block
    # 护栏必须存在：不知情的话模型会拿"用户画像"当证据答题
    assert "以资料为准" in block


def test_render_respects_char_budget():
    memories = [{"key": f"k{i}", "value": "x" * 40} for i in range(20)]
    block = render_long_term_block(memories, max_chars=100)
    body = block.split("\n", 1)[1]
    assert len(body) <= 120        # 只保留能把预算装下的那几条
    assert block.count("- ") < 20


# =====================================================================
# 4. 抽取提示词与解析
# =====================================================================
def test_extract_prompt_replaces_placeholders_without_format_error():
    # 提示词里带 JSON 示例的 {}，若用 str.format 会 KeyError——这条是回归测试
    p = build_extract_prompt("我叫小王，喜欢川菜", "好的，记住了" * 100)
    assert "__Q__" not in p and "__A__" not in p
    assert "我叫小王" in p
    assert len(p) < 4000


def test_parse_facts_normal_array():
    raw = '[{"key": "称呼", "value": "小王", "category": "profile", "confidence": 0.9}]'
    facts = parse_facts(raw)
    assert len(facts) == 1
    assert facts[0]["key"] == "称呼"
    assert facts[0]["confidence"] == 0.9


def test_parse_facts_with_fence_and_chatter():
    raw = '好的，我抽取到：\n```json\n[{"key":"口味","value":"喜欢川菜"}]\n```\n希望有帮助'
    facts = parse_facts(raw)
    assert facts and facts[0]["value"] == "喜欢川菜"
    assert facts[0]["category"] == "profile"     # 缺省分类


def test_parse_facts_single_object_and_wrapper():
    assert parse_facts('{"key":"城市","value":"深圳"}')[0]["value"] == "深圳"
    wrapped = '{"facts": [{"key":"城市","value":"深圳"}]}'
    assert parse_facts(wrapped)[0]["value"] == "深圳"


def test_parse_facts_bad_input_returns_empty():
    assert parse_facts("") == []
    assert parse_facts("我不知道该抽什么") == []
    assert parse_facts("[not json]") == []


def test_parse_facts_drops_noise_and_caps_count():
    raw = (
        '[{"key":"称呼","value":"未知"},'
        '{"key":"","value":"空键"},'
        '{"key":"' + "k" * 30 + '","value":"键太长"},'
        '{"key":"a","value":"1"},{"key":"b","value":"2"},{"key":"c","value":"3"},'
        '{"key":"d","value":"4"}]'
    )
    facts = parse_facts(raw)
    assert len(facts) == 3          # 最多 3 条
    assert all(f["value"] != "未知" for f in facts)


def test_parse_facts_coerces_confidence_and_category():
    facts = parse_facts('[{"key":"目标","value":"准备秋招","category":"GOAL","confidence":"0.85"}]')
    assert facts[0]["category"] == "goal"
    assert facts[0]["confidence"] == 0.85


# =====================================================================
# 5. 依据检查：只认用户自己说过的话
# =====================================================================
def test_evidence_support_basic():
    assert evidence_support("张三", "我叫张三，在上海做后端开发") > 0.9
    assert evidence_support("小李", "它的主人是谁？") == 0.0


def test_parse_facts_drops_facts_never_said_by_user():
    """实测 bug 的回归测试：
    用户问"它的主人是谁"，助手答"代码的主人是小李"，
    抽取器返回"称呼=小李"——这条必须被丢掉，否则会把用户自己说的名字覆盖掉。
    """
    raw = (
        '[{"key":"称呼","value":"小李"},'
        '{"key":"称呼","value":"张三"}]'
    )
    facts = parse_facts(raw, evidence="我叫张三，在上海做后端开发")
    assert [f["value"] for f in facts] == ["张三"]


def test_parse_facts_without_evidence_keeps_everything():
    raw = '[{"key":"称呼","value":"小李"}]'
    assert len(parse_facts(raw)) == 1          # 不传 evidence 时不拦，纯解析


def test_parse_facts_keeps_lightly_rephrased_value():
    # 轻微改写（"吃饭偏辣" vs "吃饭偏辣"）不该因为用词不完全一致就被丢掉
    raw = '[{"key":"口味偏好","value":"吃饭偏辣"}]'
    assert len(parse_facts(raw, evidence="我叫张三，吃饭偏辣")) == 1
