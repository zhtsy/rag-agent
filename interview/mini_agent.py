#!/usr/bin/env python3
"""
mini_agent.py — 手写 ReAct Agent（单文件 / 零第三方依赖）

按「白板上从零写」的顺序组织，七段正好对应面试官会逐段追问的七点：

  1. 工具契约 Tool：name / description / JSON Schema / 实现
  2. 注册表 REGISTRY：装饰器式注册，Agent 只认注册过的名字
  3. 参数校验 validate()：手写 JSON Schema 子集。校验失败抛 ToolError，
     由循环把错误当 Observation 回灌给模型 —— 不是抛给用户
  4. 提示词 build_prompt()：ReAct 四段式，工具清单从注册表自动生成
  5. 解析器 parse()：容错（```json 围栏、Action 名合法、Input 非 JSON）
  6. 循环 run()：max_steps 上限 / 重复动作检测 / 单步耗时告警
  7. 轨迹 Trace：把每一步的原始输出 + 观察一起返回，便于调试和演示

行数分界（面试白板只写这一部分）：
  第 1-6 节约 180 行 —— Agent 本体，必须能手写。
  第 3.5 节（三个示例工具）与第 7 节（LLM 适配 + CLI）是脚手架，
  只要求能讲清「为什么这么接」，不要求默写。

用法:
  python3 mini_agent.py                          # 离线 mock，不联网，秒回
  python3 mini_agent.py --real                   # 走本机 ollama qwen2.5:7b
  python3 mini_agent.py --real -v "自定义问题"    # -v 打印每步原始输出
  python3 mini_agent.py --question "..."         # 只换问题
"""
from __future__ import annotations

import ast
import json
import operator
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

# ============================================================
# 1. 工具契约
# ============================================================
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema，只用 type / required / properties / enum 四个关键字
    func: Callable[..., Any]


class ToolError(Exception):
    """可恢复错误：会作为 Observation 回灌给模型，让它自己改参数重试。"""


# ============================================================
# 2. 注册表
# ============================================================
REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict):
    def deco(fn):
        REGISTRY[name] = Tool(name, description, parameters, fn)
        return fn
    return deco


# ============================================================
# 3. 参数校验（JSON Schema 子集，不依赖 jsonschema 包）
# ============================================================
_PYTYPE = {"string": str, "integer": int, "number": (int, float),
           "boolean": bool, "object": dict, "array": list}


def validate(schema: dict, args: Any) -> None:
    if not isinstance(args, dict):
        raise ToolError(f"参数必须是 JSON 对象，收到 {type(args).__name__}")
    props = schema.get("properties", {})
    # 顺序有讲究：先查未知键，再查必填。模型把 "expression" 写成 "expr" 时，
    # 「未知参数 expr」比「缺少必填 expression」更直接指向病根。
    for k in args:
        if k not in props:
            raise ToolError(f"未知参数 '{k}'，可用参数：{list(props)}")
    for k in schema.get("required", []):
        if k not in args:
            raise ToolError(f"缺少必填参数 '{k}'")
    for k, v in args.items():
        t = props[k].get("type")
        py = _PYTYPE.get(t)
        ok = isinstance(v, py) if py else True
        # bool 是 int 的子类，必须显式拦掉，否则 True 能冒充 1
        if t == "integer" and isinstance(v, bool):
            ok = False
        if not ok:
            raise ToolError(f"参数 '{k}' 应为 {t}，收到 {type(v).__name__}")
        if "enum" in props[k] and v not in props[k]["enum"]:
            raise ToolError(f"参数 '{k}' 只能取 {props[k]['enum']}")


def call_tool(name: str, args: dict, slow_warn: float = 5.0) -> str:
    if name not in REGISTRY:
        raise ToolError(f"不存在工具 '{name}'，可用：{list(REGISTRY)}")
    t = REGISTRY[name]
    validate(t.parameters, args)
    t0 = time.time()
    try:
        out = t.func(**args)
    except ToolError:
        raise
    except Exception as e:                      # 工具内部崩了，也要变成可恢复错误
        raise ToolError(f"{type(e).__name__}: {e}") from e
    cost = time.time() - t0
    # 注：这是「事后告警」而不是硬超时。真要硬超时，标准库只有 signal（限主线程）
    # 或线程池 + join(timeout)，请求里说清楚，比含糊带过强。
    if cost > slow_warn:
        out = f"{out}\n[注意：该工具耗时 {cost:.1f}s，已超过 {slow_warn}s]"
    return str(out)


# ============================================================
# 3.5 三个示例工具
# ============================================================
_DOCS = {
    "D1": "评测集共 48 条：修复前答对 44 条（91.67%），修复后答对 48 条（100%）。",
    "D2": "检索层共 145 个文本块；TOP_K=5；rerank 候选 10 个；相关度阈值 0.15。",
    "D3": "图谱固定两跳：人物 -养了-> 宠物 -属于-> 品种 -属于类别-> 类别。",
    "D4": "后端只暴露 POST /chat/stream（SSE），需带 X-API-Key。",
}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg}


def _eval(node):
    """白名单求值：只允许数字常量和四则/取余/幂，杜绝 eval 注入。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ToolError("表达式只支持数字与 + - * / % **")


@tool("calc", "计算一个算术表达式，支持 + - * / % ** 和括号。",
      {"type": "object", "required": ["expression"],
       "properties": {"expression": {"type": "string"}}})
def calc(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"表达式语法错误：{e.msg}")
    v = _eval(tree.body)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else str(round(v, 4))
    return str(v)


@tool("search_docs",
      "按关键词检索本地小语料。query 请用 2-4 个字的短关键词，不要用整句话。",
      {"type": "object", "required": ["query"],
       "properties": {"query": {"type": "string"},
                      "top_k": {"type": "integer"}}})
def search_docs(query: str, top_k: int = 2) -> str:
    """字符 bigram 覆盖度打分：中文不装分词器也能跑，比「整句子串匹配」稳得多。"""
    def grams(s: str) -> set[str]:
        s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", "", s)
        return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())

    q = grams(query)
    scored = []
    for did, text in _DOCS.items():
        cover = len(q & grams(text)) / len(q) if q else 0.0
        if cover >= 0.15:
            scored.append((cover, did, text))
    if not scored:
        return "未命中。请换更短的关键词（2-4 个字）重试，例如「答对」「评测」「检索」。"
    scored.sort(reverse=True)
    return "\n".join(f"[{did}] {text}" for _, did, text in scored[:max(1, top_k)])


@tool("now", "获取当前时间。",
      {"type": "object", "required": ["fmt"],
       "properties": {"fmt": {"type": "string", "enum": ["date", "datetime"]}}})
def now(fmt: str) -> str:
    return time.strftime("%Y-%m-%d" if fmt == "date" else "%Y-%m-%d %H:%M:%S")


# ============================================================
# 4. 提示词
# ============================================================
def render_tools() -> str:
    lines = []
    for t in REGISTRY.values():
        props = t.parameters.get("properties", {})
        req = set(t.parameters.get("required", []))
        sig = ", ".join(f"{k}: {v.get('type', 'any')}" + ("" if k in req else "（可选）")
                        for k, v in props.items())
        lines.append(f"- {t.name}({sig})：{t.description}")
    return "\n".join(lines)


def build_prompt(scratchpad: str, max_steps: int) -> str:
    return f"""你可以调用工具来回答问题。

可用工具：
{render_tools()}

严格按以下格式输出，一次只输出一个 Action：

Thought: 你的推理
Action: 工具名，必须是 [{", ".join(REGISTRY)}] 之一
Action Input: 一个 JSON 对象，例如 {{"query": "关键词"}}
Observation: 系统会填这一行，你不要自己写

信息足够后，改为输出：
Thought: 信息已足够
Final Answer: 给用户的答案

硬性规则：
1. 最多 {max_steps} 步，超出会被强制终止。
2. Action Input 必须是合法 JSON 对象，键名与工具签名一致。
3. 不要臆造 Observation；工具报错就根据错误信息改参数。
4. 数值计算必须调用 calc，不要心算。

{scratchpad}"""


# ============================================================
# 5. 解析器
# ============================================================
_RE_ACTION = re.compile(r"Action\s*[:：]\s*([A-Za-z_]\w*)")
_RE_INPUT = re.compile(r"Action\s*Input\s*[:：]\s*(.*)", re.S | re.I)
_RE_FINAL = re.compile(r"Final\s*Answer\s*[:：]\s*(.*)", re.S | re.I)


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[A-Za-z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _load_json(raw: str) -> dict:
    s = _strip_fence(raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)        # 兜底：模型在 JSON 前后加了废话
        if not m:
            raise
        return json.loads(m.group(0))


def parse(text: str) -> tuple[str, str, str]:
    """返回 (kind, value, raw_input)。kind: action / final / none

    先找 Action 再找 Final Answer：模型常在 Thought 里提一句「最终答案是…」，
    反过来判会提前终止。
    """
    m_action, m_input = _RE_ACTION.search(text), _RE_INPUT.search(text)
    if m_action and m_input:
        raw = re.split(r"\n\s*Observation\s*[:：]", m_input.group(1), maxsplit=1)[0]
        return "action", m_action.group(1), raw.strip()
    m = _RE_FINAL.search(text)
    if m:
        return "final", m.group(1).strip(), ""
    return "none", "", ""


# ============================================================
# 6. 循环
# ============================================================
@dataclass
class Step:
    i: int
    output: str          # 模型原始输出
    observation: str     # 我们回灌的内容


@dataclass
class Trace:
    answer: str | None
    steps: list[Step]
    stop_reason: str     # final / max_steps


def _clip(out: str) -> str:
    """模型经常擅自续写一行 Observation（自己猜工具结果）。回灌前必须截断，
    否则 prompt 里会出现两条互相矛盾的观察值，下一步模型就懵了。"""
    return re.split(r"\n\s*Observation\s*[:：]", out, maxsplit=1)[0].rstrip()


def run(question: str, llm, max_steps: int = 6, verbose: bool = False) -> Trace:
    scratch = f"Question: {question}\n"
    steps: list[Step] = []
    recent: list[tuple] = []          # 最近两次 (tool, args)，用于查原地打转

    for i in range(1, max_steps + 1):
        out = _clip(llm(build_prompt(scratch, max_steps)).strip())
        kind, value, raw = parse(out)

        if kind == "final":
            steps.append(Step(i, out, ""))
            return Trace(value or None, steps, "final")

        if kind == "none":
            obs = "格式错误：必须包含 'Action:' + 'Action Input:'，或 'Final Answer:'。"
        else:
            try:
                args = _load_json(raw) if raw and raw != "null" else {}
            except json.JSONDecodeError as e:
                obs = f"Action Input 不是合法 JSON（{e.msg}），请输出 {{\"键\": \"值\"}}。"
                args = None
            if args is not None:
                key = (value, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if key in recent:
                    obs = "重复调用：相同参数已经调用过，请换思路或直接给 Final Answer。"
                else:
                    recent = (recent + [key])[-2:]
                    try:
                        obs = call_tool(value, args)
                    except ToolError as e:
                        obs = f"工具调用失败：{e}"

        steps.append(Step(i, out, obs))
        if verbose:
            print(f"--- step {i} ---\n{out}\n>>> Observation: {obs}")
        scratch += f"{out}\nObservation: {obs}\n"

    return Trace(None, steps, "max_steps")


# ============================================================
# 7. LLM 适配：mock 跑通逻辑，ollama 跑真实模型
# ============================================================
class MockLLM:
    """按预置脚本逐条吐输出，用来验证循环/解析/校验，不花 token、不联网。"""

    def __init__(self, script: list[str]):
        self.script, self.i = list(script), 0

    def __call__(self, prompt: str) -> str:
        if self.i >= len(self.script):
            return "Final Answer: （脚本已用尽）"
        self.i += 1
        return self.script[self.i - 1]


class OllamaLLM:
    def __init__(self, model: str = "qwen2.5:7b",
                 url: str = "http://localhost:11434/api/generate"):
        self.model, self.url = model, url

    def __call__(self, prompt: str) -> str:
        body = {"model": self.model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096},
                # stop 是关键：模型不会自己编造 Observation
                "stop": ["\nObservation:", "\nQuestion:"]}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"]


MOCK_SCRIPT = [
    'Thought: 需要先查修复前后的答对条数\n'
    'Action: search_docs\nAction Input: {"query": "修复前 修复后 答对"}',

    'Thought: 44/48 与 48/48，差值除以 48\n'
    'Action: calc\nAction Input: {"expression": "(48-44)/48*100"}',

    'Thought: 提升 8.33 个百分点\nFinal Answer: 答对率从 91.67% 提升到 100%，提升 8.33 个百分点。',
]

DEFAULT_Q = "修复后比修复前，答对率提升了几个百分点？"


# ============================================================
# main
# ============================================================
def main(argv: list[str]) -> int:
    real = "--real" in argv
    verbose = "-v" in argv or "--verbose" in argv
    q = DEFAULT_Q
    if "--question" in argv:
        q = argv[argv.index("--question") + 1]
    elif argv and not argv[-1].startswith("-") and argv[-1] not in ("--real", "-v", "--verbose"):
        q = argv[-1]

    llm = OllamaLLM() if real else MockLLM(MOCK_SCRIPT)
    t0 = time.time()
    tr = run(q, llm, max_steps=6, verbose=verbose)
    cost = time.time() - t0

    print(f"\n问题: {q}")
    print(f"模型: {'ollama/qwen2.5:7b' if real else 'mock'}   步数: {len(tr.steps)}   "
          f"终止: {tr.stop_reason}   耗时: {cost:.2f}s")
    if tr.answer:
        print(f"答案: {tr.answer}")
    else:
        # 生产上这里不能静默返回 None，要把已有 Observation 总结给用户并说明未完成
        print("答案: （未在步数内得出结论，应降级为「基于已知信息的部分回答」）")
    if not verbose:
        for s in tr.steps:
            first = s.output.strip().splitlines()[0][:60]
            print(f"  [{s.i}] {first}  -> {s.observation[:50]}")
    return 0 if tr.answer else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
