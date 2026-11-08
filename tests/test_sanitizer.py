# -*- coding: utf-8 -*-
"""sanitizer.py 的单元测试（零第三方依赖，0.1 秒内跑完）

这批函数以前长在 rag_agent.py 里，只能靠端到端评测间接覆盖 —— 意味着
一次格式回归要跑完整套评测（几分钟 + 真调模型）才能发现，发现了也不知道
是哪一条。搬出来之后可以逐条钉住。

测试样本全部来自**实际踩过的坑**，不是构造的：
- 未闭合 JSON（真流式的常态）
- `query:` / `arguments:` 被当正文吐出来
- PydanticOutputParser 的 format_instructions 被模型复述
- 正文里出现大括号被误判成结构化输出（旧 bug）

最后一条尤其重要：那是"改对了"而不是"加上了"的用例 ——
只有它能防止后人把 `_parse_structured` 改回"见大括号就当 JSON"。
"""

import pytest

from sanitizer import (
    _AnswerStreamCleaner,
    _brutal_clean,
    _extract_answer_text,
    _has_instruction_leak,
    _is_leak_prefix,
    _json_answer_prefix,
    _looks_like_partial_label,
    _looks_like_tool_call_text,
    _parse_structured,
    _parse_tool_calls_from_text,
    _strip_leading_label,
)


# ========== A. 从（可能未闭合的）JSON 里抠 answer ==========

class TestJsonAnswerPrefix:
    def test_complete_json(self):
        assert _json_answer_prefix('{"answer": "小智喜欢猫", "sources": []}') == "小智喜欢猫"

    def test_unclosed_json_still_yields_prefix(self):
        """真流式的核心要求：JSON 没闭合也要能取到当前内容，不能等。"""
        assert _json_answer_prefix('{"answer": "小智喜欢') == "小智喜欢"

    def test_escaped_quote_inside_answer(self):
        buf = '{"answer": "他说\\"好\\"", "sources": []}'
        assert _json_answer_prefix(buf) == '他说"好"'

    def test_escaped_newline(self):
        assert _json_answer_prefix('{"answer": "第一行\\n第二行"}') == "第一行\n第二行"

    def test_loose_label_form(self):
        """模型不吐 JSON、只吐 `answer: xxx` 时的兜底路径。"""
        assert _json_answer_prefix("-answer: 小智喜欢猫") == "小智喜欢猫"

    def test_no_answer_field_returns_empty(self):
        assert _json_answer_prefix('{"sources": ["a"]}') == ""

    def test_empty_input(self):
        assert _json_answer_prefix("") == ""


class TestExtractAnswerText:
    def test_plain_text_passthrough(self):
        assert _extract_answer_text("小智养了一只猫。") == "小智养了一只猫。"

    def test_json_uses_answer_field(self):
        assert _extract_answer_text('{"answer": "答案是A", "sources": ["s1"]}') == "答案是A"

    def test_strips_label_when_not_json(self):
        assert _extract_answer_text("answer: 小智喜欢猫") == "小智喜欢猫"

    def test_non_str_tolerated(self):
        assert _extract_answer_text(None) == ""
        assert _extract_answer_text(123) == "123"


class TestParseStructured:
    def test_json_with_answer(self):
        out = _parse_structured('{"answer": "A", "sources": ["s1", "s2"]}')
        assert out == {"answer": "A", "sources": ["s1", "s2"]}

    def test_braces_in_prose_are_not_json(self):
        """回归：正文里出现 {a: 1} 不能被当成结构化输出解析。

        注意断言的是"没走 JSON 分支"而不是"原样返回"——走文本兜底时会去掉大括号，
        这是原有行为（本次搬移不做改变），新的行为是它至少不会把 answer 变成 JSON 垃圾。
        """
        raw = "这个函数返回 {a: 1}，然后继续说明。"
        out = _parse_structured(raw)
        assert "这个函数返回" in out["answer"]
        assert "然后继续说明" in out["answer"]
        assert not out["answer"].startswith("{")
        assert out["sources"] == []

    def test_json_without_answer_field_falls_back_to_text(self):
        """只有 sources 没有 answer：不能把整段 JSON 当答案，要走文本兜底。"""
        out = _parse_structured('{"sources": ["a"]}')
        assert "answer" not in out["answer"]

    def test_sources_not_list_is_wrapped(self):
        out = _parse_structured('{"answer": "A", "sources": "只有一条"}')
        assert out["sources"] == ["只有一条"]

    def test_never_invents_confidence(self):
        """confidence 必须由检索证据算，解析层不许凭空造数。"""
        out = _parse_structured('{"answer": "A"}')
        assert "confidence" not in out


# ========== B. 行首标签 / 工具参数文本 ==========

class TestLeadingLabel:
    @pytest.mark.parametrize("raw,expected", [
        ("-answer: 小智喜欢猫", "小智喜欢猫"),
        ('"answer": 小智喜欢猫', "小智喜欢猫"),
        ("answer：小智喜欢猫", "小智喜欢猫"),
        ("sources: 无", "无"),
    ])
    def test_labels_are_stripped(self, raw, expected):
        assert _strip_leading_label(raw) == expected

    def test_only_first_occurrence(self):
        """正文中后续出现的 answer: 是内容的一部分，不能连带删掉。"""
        assert _strip_leading_label("answer: A answer: B") == "A answer: B"

    def test_plain_text_untouched(self):
        assert _strip_leading_label("小智喜欢猫") == "小智喜欢猫"


class TestToolCallText:
    # 注意：以 `{"` 开头的形态不归这个函数判（_TOOL_ARG_HEAD_RE 不允许首字符是 {）。
    # 纯 JSON 形态由 _parse_tool_calls_from_text 里的 '"name":' 分支负责，见 TestParseToolCalls。
    @pytest.mark.parametrize("raw", [
        "query: 小智喜欢什么",
        "quiry: 小智",
        "arguments: {}",
        "parameters: {}",
        'search_knowledge_base({"query": "x"})',
        "<|python_tag|>search_knowledge_base",
    ])
    def test_detects_dirty_shapes(self, raw):
        assert _looks_like_tool_call_text(raw) is True

    @pytest.mark.parametrize("raw", [
        "小智喜欢猫。",
        "答案是 A。",
        "这是一个查询问题。",
        "",
    ])
    def test_normal_answer_not_flagged(self, raw):
        assert _looks_like_tool_call_text(raw) is False


class TestPartialLabel:
    def test_single_char_prefix_is_held(self):
        """'q' 这种 1 字符块必须挂起，否则 'uery:' 会跟着漏出去。"""
        assert _looks_like_partial_label("q") is True
        assert _looks_like_partial_label("-answ") is True
        assert _looks_like_partial_label("query") is True

    def test_complete_word_with_colon_is_content(self):
        assert _looks_like_partial_label("答案：小智喜欢猫") is False


# ========== C. 格式指令泄漏 ==========

class TestInstructionLeak:
    def test_detects_english_format_instructions(self):
        raw = ("The output should be formatted as a JSON instance that conforms "
               "to the JSON schema below")
        assert _has_instruction_leak(raw) is True

    def test_detects_chinese_format_requirement(self):
        assert _has_instruction_leak("输出格式要求：请严格按照如下格式") is True

    def test_detects_with_broken_spacing(self):
        """流式切块会把空格切乱，_norm_probe 要能折叠回来。"""
        assert _has_instruction_leak("The output   should  be formatted") is True

    def test_normal_answer_not_flagged(self):
        assert _has_instruction_leak("小智喜欢猫，它有三年了。") is False

    def test_prefix_is_held_not_killed(self):
        """前半截应该"挂起等待"，不是"判死"——判死会吃掉正常答案。"""
        assert _is_leak_prefix("The output") is True
        assert _has_instruction_leak("The output") is False


# ========== D. 从脏文本还原 tool_calls ==========

class TestParseToolCalls:
    def _strip_ids(self, calls):
        return [(c["name"], c["args"]) for c in calls]

    def test_query_only_object_is_not_parsed(self):
        """已知盲区（只记录，不修）：纯 {"query": ...} 形态进不来。

        _parse_tool_calls_from_text 的入口只放行「_looks_like_tool_call_text 为真」
        或「含 '"name":'」的文本，而前者不允许首字符是 `{`。于是 _calls_from_json_obj
        里"只有 query 没有 name"那条分支实际上不可达。
        搬移任务不改变行为，所以这里照实钉住；要修请单独做，并同步更新此用例。
        """
        assert _parse_tool_calls_from_text('{"query": "小智喜欢什么"}') == []

    def test_standard_function_shape(self):
        raw = '{"name": "search_knowledge_base", "arguments": {"query": "猫"}}'
        calls = _parse_tool_calls_from_text(raw)
        assert self._strip_ids(calls) == [("search_knowledge_base", {"query": "猫"})]

    def test_stringified_arguments(self):
        raw = '{"function_call": {"name": "search_knowledge_base", "arguments": "{\\"query\\": \\"猫\\"}"}}'
        calls = _parse_tool_calls_from_text(raw)
        assert self._strip_ids(calls) == [("search_knowledge_base", {"query": "猫"})]

    def test_bare_query_label(self):
        calls = _parse_tool_calls_from_text("query: 泰拉瑞亚怎么打加里奥")
        assert self._strip_ids(calls) == [("search_knowledge_base", {"query": "泰拉瑞亚怎么打加里奥"})]

    def test_normal_answer_yields_no_calls(self):
        """最重要的一条：正常回答绝不能被误判成工具调用，否则会白查一轮。"""
        assert _parse_tool_calls_from_text("小智养了一只叫球球的猫。") == []

    def test_unknown_tool_name_rejected(self):
        assert _parse_tool_calls_from_text('{"name": "rm_rf", "args": {"path": "/"}}') == []

    def test_ids_are_unique(self):
        """多轮里重复的 tool_call_id 会让部分模型 API 直接报错。"""
        a = _parse_tool_calls_from_text("query: x")
        b = _parse_tool_calls_from_text("query: x")
        assert a and b
        assert a[0]["id"] != b[0]["id"]


# ========== E. 流式清洗器（有状态，最容易出错的部件） ==========

def drain(cleaner, pieces):
    buf = [cleaner.feed(p) for p in pieces]
    buf.append(cleaner.flush())
    return "".join(buf)


class TestAnswerStreamCleaner:
    def test_plain_text_passes_through(self):
        out = drain(_AnswerStreamCleaner(), ["小智", "喜欢", "猫。"])
        assert out == "小智喜欢猫。"

    def test_json_mode_emits_only_answer_field(self):
        """前端绝不能看到 `{"answer": "` 这些脚手架。"""
        pieces = ['{"ans', 'wer": "小智喜', '欢猫", "sources": []}']
        out = drain(_AnswerStreamCleaner(), pieces)
        assert out == "小智喜欢猫"

    def test_tool_call_turn_is_silenced(self):
        out = drain(_AnswerStreamCleaner(), ["query", ": 小智", "喜欢什么"])
        assert out == ""

    def test_instruction_leak_is_silenced(self):
        pieces = ["The output should ", "be formatted as a JSON instance"]
        out = drain(_AnswerStreamCleaner(), pieces)
        assert out == ""

    def test_partial_label_is_held_then_released(self):
        """'answer' 是标签就剥掉；但如果后面接的是正文，必须放出来。"""
        c = _AnswerStreamCleaner()
        assert c.feed("-answ") == ""          # 半截，挂起
        assert c.feed("er: 你好") == "你好"    # 判定为标签，剥掉后吐正文

    def test_answer_containing_query_word_is_not_eaten(self):
        """回归：答案里出现 query 这个英文词，不能被当成工具参数吞掉。

        query 之后必须紧跟冒号或大括号才判脏（见 _TOOL_ARG_HEAD_RE）；
        单纯出现这个词属于正常内容，一个字都不能少。
        """
        out = drain(_AnswerStreamCleaner(), ["query 这个词", "是指检索词。"])
        assert out == "query 这个词是指检索词。"
        out2 = drain(_AnswerStreamCleaner(), ["这个 query ", "是指检索词。"])
        assert out2 == "这个 query 是指检索词。"

    def test_label_stripped_only_once(self):
        c = _AnswerStreamCleaner()
        out = drain(c, ["answer: A", " 然后 answer: B"])
        assert out == "A 然后 answer: B"

    def test_chunking_granularity_does_not_change_output(self):
        """流式系统最典型的 bug：答案随网络切块方式变化。这里穷举切法。"""
        text = '{"answer": "小智喜欢猫，也喜欢编程。", "sources": []}'
        expected = None
        for size in range(1, 9):
            pieces = [text[i:i + size] for i in range(0, len(text), size)]
            out = drain(_AnswerStreamCleaner(), pieces)
            if expected is None:
                expected = out
            assert out == expected, f"切块大小={size} 时输出不一致: {out!r} != {expected!r}"
        assert expected == "小智喜欢猫，也喜欢编程。"

    def test_poison_stops_everything_after(self):
        c = _AnswerStreamCleaner()
        c.feed('{"query"')
        assert c.feed(": 小智") == ""
        assert c.flush() == ""


# ========== F. 最后一道暴力清洗 ==========

class TestBrutalClean:
    def test_strips_code_fence_and_query_label(self):
        raw = '```json\nquery: 小智\n```'
        assert _brutal_clean(raw).strip() == "小智"

    def test_removes_leak_line_keeps_rest(self):
        raw = "小智喜欢猫。\nThe output should be formatted as JSON\n它有三年了。"
        out = _brutal_clean(raw)
        assert "小智喜欢猫。" in out
        assert "它有三年了。" in out
        assert "The output should be" not in out

    def test_normal_text_untouched(self):
        raw = "小智养了一只叫球球的猫，今年三岁。"
        assert _brutal_clean(raw) == raw

    def test_empty(self):
        assert _brutal_clean("") == ""
