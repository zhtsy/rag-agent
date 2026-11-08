# -*- coding: utf-8 -*-
"""
sanitizer.py —— 大模型输出文本的解析与清洗

这个模块里没有任何 RAG 业务。它对抗的是另一件事：**模型不按你要求的格式输出**。

原来这近 500 行长在 rag_agent.py 里，和检索、编排、提示词混在一起。
后果是：想改检索逻辑，得先穿过一整片正则；没人敢碰的部分，就等于不存在。

搬出来的两个收益：
1. rag_agent.py 只剩编排，业务改动不必穿过这片雷区；
2. 本模块零第三方依赖（只 import re / json / uuid 三个标准库），
   所以 tests/test_sanitizer.py 能在 0.2 秒内跑完 —— 而 import rag_agent
   要加载 torch、chroma 和两个重排/向量模型。这批函数以前只能靠端到端评测
   间接覆盖，现在可以逐条定位。

⚠️ 搬运原则：逻辑一行未改，只改位置。
函数名保留了下划线前缀，是为了让 rag_agent.py 里 100+ 个调用点不必改动 ——
一旦改名，"纯搬移"就变成"重构"，风险立刻上升一个量级。
后续若要重命名，请单独做一次 commit，并确保 pytest 全绿。
"""

import json
import re
import uuid

# loguru 在容器里可用、在 CI 里没有。这里降级到标准库而不是直接 import，
# 否则本模块就不能被那套 0.2 秒的 CI 覆盖了。
try:
    from loguru import logger
except ImportError:  # pragma: no cover - 仅 CI / 裸宿主机走这里
    import logging

    logger = logging.getLogger(__name__)


# ========== A. 从（可能未闭合的）JSON 里抠 answer ==========

_ANSWER_KEY_RE = re.compile(r'"answer"\s*:\s*"')
_ANSWER_LOOSE_RE = re.compile(r'[-*#>\s]{0,3}"?answer"?\s*[:：]\s*"?', re.IGNORECASE)
_ANSWER_LABEL_RE = re.compile(r'[-*#>\s]{0,3}"?\s*(?:answer|sources|confidence|contexts)\s*"?\s*[:：]\s*', re.IGNORECASE)

_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f'}


def _json_answer_prefix(buf: str) -> str:
    """
    从一个"正在流式生成中、尚未闭合"的 JSON 字符串里，增量地抠出 answer 字段当前内容。
    这是真流式的关键：不用等 JSON 闭合，边生成边取。
    """
    if not buf:
        return ""

    m = _ANSWER_KEY_RE.search(buf)
    if m:
        rest = buf[m.end():]
        out = []
        esc = False
        for ch in rest:
            if esc:
                out.append(_ESCAPES.get(ch, ch))
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                break
            else:
                out.append(ch)
        return "".join(out)

    # 宽松兜底：-answer: xxxx 这类脏格式
    m2 = _ANSWER_LOOSE_RE.search(buf)
    if m2:
        rest = buf[m2.end():]
        stop = len(rest)
        for i, ch in enumerate(rest):
            if ch in "\n\r" or ch in "}]":
                stop = i
                break
        return rest[:stop].strip().strip('"').strip(",").strip()

    return ""


def _extract_answer_text(content: str) -> str:
    """从一段完整的（可能是 JSON 的）模型输出里取出纯答案文本"""
    if not content:
        return ""
    if not isinstance(content, str):
        return str(content)

    text = _json_answer_prefix(content)
    if text.strip():
        return text.strip()

    cleaned = content.replace("{", "").replace("}", "")
    cleaned = _ANSWER_LABEL_RE.sub("", cleaned)
    return cleaned.strip()


def _parse_structured(content: str) -> dict:
    """把最终 AI 消息解析成 {answer, sources}。

    只认「真的带了 answer 字段」的 JSON；否则一律按纯文本处理。
    原来的写法是「正文里出现一对大括号就当 JSON」，正文随便引用一段代码/公式
    会被误当成结构化输出，answer 直接变成垃圾。

    confidence 不在这里产生：它必须由检索证据统一算出（见 _compute_confidence），
    任何"解析失败就给 0.8 / 0.5"的兜底都是凭空造数，属于同一类问题的两个变种。
    """
    data = {}
    if isinstance(content, str):
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("answer"):
                data = parsed

    answer = data.get("answer") or _extract_answer_text(content)
    sources = data.get("sources") or []
    if not isinstance(sources, list):
        sources = [str(sources)]

    return {"answer": answer, "sources": sources}


# ========== B. 行首标签 / 工具参数文本的识别 ==========

# 行首标签：-answer: / "sources": 之类，出现在正文最开头时需要剥掉
_LEADING_LABEL_RE = re.compile(
    r'^\s*[-*#>\s]{0,3}"?\s*(?:answer|sources|confidence|contexts)\s*"?\s*[:：]\s*"?',
    re.IGNORECASE,
)
# 把工具参数的标签也纳入"半截标签"识别范围。
# 否则 'q' 这种 1 个字符的 chunk 会被立刻当成正文吐出去，随后的 'uery:' 就跟出去了。
_LABEL_WORDS = (
    "answer", "sources", "confidence", "contexts",
    "query", "quiry", "qurey", "qury", "arguments", "parameters", "params", "name",
)

# 行首形态：query: / quiry: / arguments: / parameters: / name: / {"query"...
_TOOL_ARG_HEAD_RE = re.compile(
    r'^\s*[-*#>\s"\'`]{0,3}"?\s*'
    r'(?:query|quiry|qurey|qury|question|arguments?|args|params|parameters?'
    r'|tool_name|name|function|tool_calls?)\s*["\']?\s*[:：{\[]',
    re.IGNORECASE,
)
# 函数调用语法 / 特殊 token
_TOOL_CALL_SYNTAX_RE = re.compile(
    r'<\|python_tag\|>|(?:search_knowledge_base|search_knowledge_graph)\s*[\(\{]',
    re.IGNORECASE,
)
# 从脏文本里抠 query 值
_QUERY_VALUE_RE = re.compile(
    r'(?:query|quiry|qurey|qury|question|检索词|问题)\s*["\']?\s*[:：=]\s*["\']?([^"\'，,}\n]+)',
    re.IGNORECASE,
)
_VALID_TOOL_NAMES = {"search_knowledge_base", "search_knowledge_graph"}

# 探测期需要"挂起等待"的词：结构化标签 + 工具函数名
_HOLD_WORDS = _LABEL_WORDS + ("search_knowledge_base", "search_knowledge_graph")


def _strip_leading_label(text: str) -> str:
    """若大模型吐出的第一段正文中有-answer:、"answer":这类标签，精确的切割"""
    return _LEADING_LABEL_RE.sub("", text, count=1)


def _looks_like_partial_label(text: str) -> bool:
    """判断 text 是不是"某个结构化标签的前半截"（例如 '-answer'、'query'）"""
    t = re.sub(r'^[-*#>\s]*"?\s*', "", text).lower()
    if not t:
        return True
    for w in _HOLD_WORDS:
        if w.startswith(t):
            return True
        if t.startswith(w) and len(t) <= len(w) + 3 and ":" not in t and "：" not in t:
            return True
    return False


def _looks_like_tool_call_text(text: str) -> bool:
    """判断一段模型输出是不是"工具调用的参数文本" """
    head = (text or "")[:80]
    if not head:
        return False
    return bool(_TOOL_ARG_HEAD_RE.match(head) or _TOOL_CALL_SYNTAX_RE.search(head))


# ========== C. 格式指令泄漏（模型把 format_instructions 当正文吐出来） ==========
# PydanticOutputParser 注入的那段英文说明，Qwen 7B 会原样复述：
#   "The output should be formatted as a JSON instance that conforms to the JSON schema below..."
# 这类文本一个字都不能到前端。
_INSTRUCTION_LEAK_RE = re.compile(
    r"the\s+output\s+should\s+be\s+formatted"
    r"|json\s+instance"
    r"|json\s+schema"
    r"|here\s+is\s+the\s+output\s+schema"
    r"|wrap\s+the\s+output"
    r"|use\s+the\s+following\s+pydantic"
    r"|output\s+should\s+conform"
    r"|格式化(?:为|成).{0,8}json"
    r"|输出(?:格式|要求)"
    r"|请(?:严格)?按(?:照)?(?:如下|以下)格式",
    re.IGNORECASE,
)

# 探测期需要"挂起"的泄漏前缀：流式切块会把 "The output should be" 切成
# "The" / " outp" / "ut should"...，必须等看清了再决定，否则第一块就漏出去。
_LEAK_PREFIXES = (
    "the output should be formatted",
    "the output should be",
    "the output",
    "here is the output",
    "json instance",
    "json schema",
    "wrap the output",
    "use the following pydantic",
    "output should conform",
)

# 最后一道保险：逐行删掉疑似"格式说明"的整行
_INSTRUCTION_LEAK_LINE_RE = re.compile(
    r"^.*(?:the output should be|json instance|json schema|here is the output"
    r"|wrap the output|use the following pydantic|输出格式要求|请严格按照如下格式).*$",
    re.IGNORECASE | re.MULTILINE,
)


def _norm_probe(text: str) -> str:
    """小写 + 折叠空白：流式切块会把 'The output  should' 切得七零八落"""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _has_instruction_leak(text: str) -> bool:
    """完整命中泄漏特征 —— 该判死了"""
    return bool(text) and bool(_INSTRUCTION_LEAK_RE.search(_norm_probe(text)))


def _is_leak_prefix(text: str) -> bool:
    """只是泄漏句子的前半截（'the output'）—— 先挂起别发"""
    t = _norm_probe(text)
    if not t:
        return False
    return any(p.startswith(t) for p in _LEAK_PREFIXES)


# ========== D. 从脏文本还原 tool_calls ==========

def _make_tool_call(name: str, args: dict) -> dict:
    """将工具名和参数，打包组装成一个带有唯一随机 ID、符合大模型标准 API 规范的“工具调用指令”字典。"""
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}


def _iter_json_objects(text: str):
    """从任意脏文本里迭代出所有能解析的 JSON 对象"""
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            return
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            yield obj
        idx = start + end


def _calls_from_json_obj(obj: dict) -> list:
    """把各种"工具调用形状"的 dict 归一成 tool_calls"""
    if not isinstance(obj, dict):
        return []

    for key in ("tool_calls", "tool_call", "function_call"):
        if key in obj:
            val = obj[key]
            items = val if isinstance(val, list) else [val]
            out = []
            for item in items:
                out.extend(_calls_from_json_obj(item))
            return out

    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    func = obj.get("function")
    args = {}
    if isinstance(func, dict):                       # {"function": {"name": ..., "arguments": ...}}
        name = name or func.get("name")
        args = func.get("arguments") or func.get("parameters") or {}
    elif isinstance(func, str) and not name:
        name = func
    if not args:
        args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or obj.get("input") or {}

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"query": args}
    if not isinstance(args, dict):
        args = {}

    # 只有 query 没有 name 的形态：{"query": "小智喜欢什么"}
    if not name and obj.get("query"):
        name, args = "search_knowledge_base", {"query": obj["query"]}
    if not name and args.get("query"):
        name = "search_knowledge_base"

    if name in _VALID_TOOL_NAMES and args:
        return [_make_tool_call(name, args)]
    return []


def _parse_tool_calls_from_text(content: str) -> list:
    """
    Qwen 7B 不支持标准流式工具调用时，会把工具参数当正文吐出来
    （例如 'query: 小智喜欢什么' / '{"name": "search_knowledge_base", ...}'）。
    这里做一次兜底：把文本还原成 tool_calls，让图照常走 tools → agent 的流程。
    """
    text = (content or "").strip()
    if not text:
        return []
    # 只有"看起来像工具调用"才解析，避免把正常回答误判成工具调用
    if not (_looks_like_tool_call_text(text) or re.search(r'"name"\s*:', text)):
        return []

    calls = []
    for obj in _iter_json_objects(text):
        calls.extend(_calls_from_json_obj(obj))
    if calls:
        return calls

    # 最后兜底：直接从 "query: xxx" 里抠出检索词
    m = _QUERY_VALUE_RE.search(text)
    if m:
        q = m.group(1).strip().strip('"\'')
        if q:
            return [_make_tool_call("search_knowledge_base", {"query": q})]
    return []


# ========== E. 流式 token 清洗器 ==========

class _AnswerStreamCleaner:
    """
    流式 token 清洗器（每个 LLM run 一个实例）。

    Qwen 走 bind_tools + JSON 结构化输出，流的 token 里混着 `{"`、`"answer": "`、`-answer:`
    这些脚手架；工具调用那轮的参数也是以 JSON 形式流出来的，绝不能让它们漏到前端。

    状态机：
    1. 探测期：攒几个字符再判断形状
       - 以 { 或 [ 开头 → JSON 模式（只吐 answer 字段内容）
       - 像工具参数     → poison，本轮一个字都不输出
       - 其它           → 纯文本模式
    2. 识别出 tool_call_chunks（工具轮）→ poison
    3. flush()：收尾，纯文本模式把探测期残留吐完
    """

    PROBE_CHARS = 6
    # 48 而不是 24："The output should be formatted" 就要 31 个字符才看得清，
    # 挂起窗口太短会在半截处误判成正文放出去。
    MAX_HOLD = 48

    def __init__(self):
        self._mode = None          # None=探测中 / "text" / "json" / "discard"
        self._probe = ""
        self._json = ""
        self._emitted_len = 0
        self._first_text = True    # 正文第一段需要剥掉行首的 -answer: 之类的标签

    def poison(self):
        """标记为工具调用轮：本轮后续所有 token 全部丢弃"""
        self._mode = "discard"
        self._probe = ""
        self._json = ""

    def _emit_text(self, text: str) -> str:
        if not text:
            return ""
        if self._first_text:
            self._first_text = False
            return _strip_leading_label(text)
        return text

    def feed(self, piece: str) -> str:
        if not piece:
            return ""
        if self._mode == "discard":
            return ""
        if self._mode == "text":
            return piece

        if self._mode is None:
            self._probe += piece
            stripped = self._probe.lstrip()
            if not stripped:
                return ""
            if stripped[0] in "{[":
                self._mode = "json"
                self._json = self._probe
                self._probe = ""
                return self._delta()
            # Qwen 7B 会把 query:/arguments:/name:{ 这类工具参数当正文吐出来
            if _looks_like_tool_call_text(stripped):
                logger.debug(f"🛑 探测到工具参数文本，判定为脏数据: {stripped[:40]!r}")
                self.poison()
                return ""
            # 模型复述了 format_instructions（The output should be formatted as a JSON instance...）
            if _has_instruction_leak(stripped):
                logger.debug(f"🛑 探测到格式指令泄漏，判定为脏数据: {stripped[:60]!r}")
                self.poison()
                return ""
            # 只是泄漏句子的前半截（"The outpu"）→ 先挂起，别急着发
            if _is_leak_prefix(stripped) and len(self._probe) < self.MAX_HOLD:
                return ""
            # 半截标签先挂起，等情况明朗再发
            if _looks_like_partial_label(stripped) and len(self._probe) < self.MAX_HOLD:
                return ""
            self._mode = "text"
            out = self._probe
            self._probe = ""
            return self._emit_text(out)

        # JSON 模式
        self._json += piece
        return self._delta()

    def _delta(self) -> str:
        text = _json_answer_prefix(self._json)
        if len(text) <= self._emitted_len:
            return ""
        seg = text[self._emitted_len:]
        self._emitted_len = len(text)
        return seg

    def flush(self) -> str:
        if self._mode is None:
            out = self._probe
            self._probe = ""
            self._mode = "text"
            # 收尾时再判一次：整轮就吐了个 "query:xxx" 的，别放出去
            if _looks_like_tool_call_text(out):
                self.poison()
                return ""
            # 整轮就吐了半句格式说明的，也别放出去
            if _has_instruction_leak(out) or _is_leak_prefix(out):
                logger.debug(f"🛑 收尾检测到格式指令残留: {out[:60]!r}")
                self.poison()
                return ""
            return self._emit_text(out)
        return ""


# ========== F. 最后一道暴力清洗 ==========

def _brutal_clean(text: str) -> str:
    """
    【核心修复 D】暴力清洗：不管三七二十一，把模型漏出来的脚手架全删掉。
    后端清洗器已经保证是纯文本，这里是最后一道保险。
    """
    if not text:
        return ""
    text = (
        text.replace("```json", "")
        .replace("```", "")
        .replace("query:", "")
        .replace("quiry:", "")
    )
    # 最后一道保险：逐行删掉复述出来的"格式说明"（可能被切成多块漏到文本模式）
    text = _INSTRUCTION_LEAK_LINE_RE.sub("", text)
    return text
