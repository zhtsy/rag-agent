# -*- coding: utf-8 -*-
"""memory_context 纯函数测试（不连库、不加载模型）。

这些函数是记忆链路里最容易出错的部分：
  · 截断切在半个句子里，下一轮模型就"忘了"关键限定
  · 摘要把实体名吃掉，下一轮"它几岁"就再也指代不了
  · 摘要渲染漏了"只作背景"的护栏，模型会拿旧结论当本轮证据
"""

import pytest

from memory_context import (
    DEFAULT_HISTORY_TURNS,
    build_summary_prompt,
    ensure_summary_entities,
    history_turns_for,
    render_summary_block,
    resolve_session_id,
    smart_truncate,
)


# =====================================================================
# 0. 历史窗口按 provider 收回
# =====================================================================
def test_window_is_wide_on_cloud_providers():
    """云端 128K 上下文，不该还按本地 7B 的 3 轮跑。"""
    assert history_turns_for("deepseek") == 16


def test_hybrid_window_is_moderate():
    """hybrid 的生成走本地 7B（num_ctx=4096），不能按云端 128K 给 16 轮。"""
    assert history_turns_for("hybrid") == 8
    assert history_turns_for("hybrid") < history_turns_for("deepseek")


def test_window_shrinks_when_falling_back_to_local():
    """降级到纯本地必须自动收回，否则检索证据被历史挤掉。"""
    assert history_turns_for("local") == 3
    assert history_turns_for("  LOCAL  ") == 3        # 大小写/空格都要能识别


def test_window_default_for_unknown_provider():
    assert history_turns_for("") == DEFAULT_HISTORY_TURNS
    assert history_turns_for("some-new-api") == DEFAULT_HISTORY_TURNS


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
# 3. 摘要提示词
# =====================================================================
_TURNS = [
    {"turn_no": 1, "question": "代码是什么品种？", "answer": "代码是一只橘猫，主人是小李。"},
    {"turn_no": 2, "question": "它几岁？", "answer": "代码今年 3 岁。"},
]


def test_summary_prompt_carries_prev_and_turns():
    p = build_summary_prompt("上一版摘要：聊过拿铁", _TURNS)
    assert "上一版摘要：聊过拿铁" in p
    assert "代码是一只橘猫" in p
    assert "[第 1 轮]" in p and "[第 2 轮]" in p


def test_summary_prompt_has_no_format_placeholders_left():
    """提示词里有大括号风险，必须是 replace 而不是 format。"""
    p = build_summary_prompt("", _TURNS)
    assert "__PREV__" not in p and "__TURNS__" not in p and "__LIMIT__" not in p


def test_summary_prompt_with_empty_inputs_is_safe():
    p = build_summary_prompt("", [])
    assert "（无）" in p


def test_summary_prompt_truncates_long_answer():
    long_turn = {"turn_no": 9, "question": "问" * 300, "answer": "答" * 2000}
    p = build_summary_prompt("", [long_turn])
    # 摘要是压缩，不是誊抄：超长回答必须被截断，否则摘要会比原文还长
    assert len(p) < 1200


# =====================================================================
# 4. 实体名兜底（摘要最致命的信息损失）
# =====================================================================
def test_entities_are_appended_when_summary_dropped_them():
    summary = "用户询问了一只宠物的品种和年龄。"
    source = "问：代码是什么品种？答：代码是一只橘猫。"
    out = ensure_summary_entities(summary, source, ["代码", "拿铁"])
    assert "代码" in out
    assert "【涉及实体】" in out
    assert "拿铁" not in out          # 原文里没有的实体不能凭空补


def test_entities_not_duplicated_when_summary_kept_them():
    summary = "代码是一只橘猫，今年 3 岁。"
    source = "问：代码是什么品种？答：代码是一只橘猫。"
    out = ensure_summary_entities(summary, source, ["代码"])
    assert out == summary             # 已经有了就不该再追加一行


def test_entities_noop_when_source_empty():
    assert ensure_summary_entities("摘要", "", ["代码"]) == "摘要"


def test_entities_multiple_missing_joined():
    out = ensure_summary_entities("聊了两只宠物。", "代码和拿铁都在客厅。", ["代码", "拿铁"])
    assert "代码" in out and "拿铁" in out


# =====================================================================
# 5. 摘要渲染
# =====================================================================
def test_render_summary_empty():
    assert render_summary_block("") == ""
    assert render_summary_block("   ") == ""


def test_render_summary_contains_guard_sentence():
    block = render_summary_block("代码是一只橘猫。")
    assert "更早的对话摘要" in block
    assert "不是本轮问题的证据" in block     # 护栏句：防止模型拿旧结论当证据
    assert "代码是一只橘猫。" in block


def test_render_summary_respects_char_budget():
    block = render_summary_block("橘猫" * 500, max_chars=100)
    assert len(block) < 260
