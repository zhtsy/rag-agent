#!/usr/bin/env python3
"""
mini_agent_bindtools.py —— 同一个 Agent，改用 function calling（bind_tools）

它和 mini_agent_interview.py 干的是完全同一件事，只是"沟通方式"换了：

  ┌────────────┬──────────────────────────┬──────────────────────────┐
  │            │ ReAct 版（手写协议）      │ bind_tools 版（本文件）   │
  ├────────────┼──────────────────────────┼──────────────────────────┤
  │ 工具清单    │ 手写进 prompt 的一行文字  │ 框架从函数签名自动生成     │
  │ 模型怎么回   │ 回一段文本，里面有 Action │ 回结构化 tool_calls 字段   │
  │ 怎么解析     │ 我用正则抠               │ 直接读 resp.tool_calls    │
  │ 怎么防编造   │ stop=["\\nObservation:"]  │ 不需要，结构化输出编不了   │
  │ 结果怎么回灌  │ 拼成 "Observation: xx"   │ ToolMessage(tool_call_id) │
  │ 工具表       │ TOOLS = {"calc": calc}   │ 还在，但是推导出来的       │
  │ 循环         │ 我写                     │ 还是我写                 │
  │ 执行工具     │ 我执行                   │ 还是我执行               │
  └────────────┴──────────────────────────┴──────────────────────────┘

一句话：bind_tools 替代的是「沟通」，不是「执行」。

跑法：
    python mini_agent_bindtools.py            # 离线 mock，秒回，不花钱
    python mini_agent_bindtools.py --real     # 真走 qwen2.5:7b
"""
import sys

# ---------- 工具：这次用框架的装饰器，不手写 schema ----------
DOCS = {
    "D1": "评测集共 48 条：修复前答对 44 条（91.67%），修复后答对 48 条（100%）。",
    "D2": "检索层共 145 个文本块；TOP_K=5；rerank 候选 10 个；相关度阈值 0.15。",
    "D3": "图谱固定两跳：人物 -养了-> 宠物 -属于-> 品种 -属于类别-> 类别。",
}

_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
        "*": lambda a, b: a * b, "/": lambda a, b: a / b}


def _eval(expr: str) -> float:
    """只认「数字 运算符 数字」，别的字符一律拒绝（不用 eval，防命令注入）。"""
    s = expr.replace(" ", "")
    for op in ("+", "-", "*", "/"):
        if op in s[1:]:                       # s[1:] 是为了跳过开头的负号
            i = s.index(op, 1)
            return _OPS[op](float(s[:i]), float(s[i + 1:]))
    return float(s)


def _bigrams(text: str) -> set:
    """中文没空格，按字符两两切片当关键词（这是真踩过的坑）。"""
    t = "".join(ch for ch in text if ch.isalnum())
    return {t[i:i + 2] for i in range(len(t) - 1)}


# ============================================================
# 关键差别 ①：工具定义
#   ReAct 版：普通函数 + 手写 TOOLS 字典 + 手写 schema
#   这一版  ：@tool 装饰器自动做三件事
#              1. 函数名  -> 工具名
#              2. docstring -> description（模型靠这句判断该不该调）
#              3. 类型标注 -> 参数 schema
# ============================================================
from langchain_core.tools import tool


@tool
def calc(expression: str) -> str:
    """计算一个只含数字和 + - * / 的算式，例如 "100 - 91.67"。

    只传算式本身，不要带单位、中文或等号。想算百分比请自己先算好差值。
    """
    return str(round(_eval(expression), 4))


@tool
def search(query: str) -> str:
    """检索本项目的资料库，返回命中的原文片段。

    资料库里存着这些事实：评测结果（修复前后各答对多少条、准确率百分比）、
    检索配置（文本块数量、TOP_K、rerank 候选数、相关度阈值）、
    图谱结构（节点标签和边类型）。

    凡是问"这个项目某个具体数字是多少"的，都先来这里查，不要凭印象回答，
    也不要让用户补充数据——数据就在库里。

    query 用短关键词（如 "答对"、"文本块"、"阈值"），不要写整句问句。
    """
    q = _bigrams(query)
    scored = []
    for did, text in DOCS.items():
        hits = len(q & _bigrams(text))        # 字符 bigram 覆盖度
        if hits:
            scored.append((hits, did, text))
    if not scored:
        return "没有命中任何文档，换个更短的关键词试试。"
    scored.sort(reverse=True)
    return "\n".join(f"[{did}] {text}" for _, did, text in scored[:1])


tools = [calc, search]

# ============================================================
# 关键差别 ②：工具表还在，只是自动生成
#   很多人以为用了 bind_tools 就不用维护这张表了——不是。
#   模型只负责说"我要调 calc"，真正按名字找到 calc 的仍是你的代码。
# ============================================================
tool_map = {t.name: t for t in tools}


# ============================================================
# 关键差别 ③：bind_tools 不替代 system prompt
#   实测踩到的坑：只给一句问题 + bind_tools，qwen2.5:7b 一个工具都不调，
#   直接反问"请提供准确率数据"。因为 schema 只说明「能调什么」，
#   不说明「什么时候该调」。这半句必须自己写。
# ============================================================
SYSTEM = """你是一个可以调用工具的助手。

规则：
1. 涉及具体数字、配置、阈值等事实时，必须先用 search 查，不许凭空猜。
2. 涉及算数时，必须用 calc，不许心算。
3. 信息不足时，先查再问，不要一上来就让用户补充数据。
4. 查到足够信息后直接给结论，不要罗列你查了什么。
"""


# ================= 核心：bind_tools 版循环（约 25 行）=================
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


def react_bindtools(question, model, max_steps=6, verbose=False):
    msgs = [SystemMessage(SYSTEM), HumanMessage(question)]  # 消息列表，不是长字符串

    for step in range(1, max_steps + 1):
        resp = model.invoke(msgs)             # model 已经 bind_tools 过
        msgs.append(resp)                     # 它的话必须进历史（下面要配对 id）

        calls = resp.tool_calls or []         # 结构化字段，不用正则
        if not calls:                         # 没有工具调用 = 它答完了
            return resp.content, step

        for call in calls:                    # 一轮可能要调好几个工具
            name, args = call["name"], call["args"]
            try:
                if name not in tool_map:      # 白名单，边界仍然在我手里
                    obs = f"没有工具 {name}，可用：{list(tool_map)}"
                else:
                    obs = str(tool_map[name].invoke(args))
            except Exception as e:            # 参数错/工具崩，都变成可回灌的文字
                obs = f"工具报错：{type(e).__name__}: {e}"

            if verbose:
                print(f"  [{step}] {name}({args}) -> {obs[:70]}")

            # 关键：tool_call_id 必须和上面那条 AIMessage 对得上，否则框架报错
            msgs.append(ToolMessage(content=obs, tool_call_id=call["id"]))

    return None, max_steps
# ================= 核心结束 =================


# ---------- 两个"模型"：一个假的（离线测循环），一个真的 ----------
from langchain_core.messages import AIMessage


class MockModel:
    """假模型：脚本里写字符串 = 直接作答，写列表 = 假装要调工具。"""

    def __init__(self, script):
        self.script, self.i = list(script), 0

    def invoke(self, msgs):
        item = self.script[self.i] if self.i < len(self.script) else "算完了。"
        self.i += 1
        if isinstance(item, str):
            return AIMessage(content=item)
        return AIMessage(content="", tool_calls=[
            {**c, "type": "tool_call", "id": f"call_{self.i}_{j}"}
            for j, c in enumerate(item)
        ])


def real_model():
    from langchain_ollama import ChatOllama
    return ChatOllama(model="qwen2.5:7b", temperature=0).bind_tools(tools)


QUESTION = "修复后的准确率比修复前提升几个百分点？"


def main():
    real = "--real" in sys.argv
    model = real_model() if real else MockModel([
        [{"name": "search", "args": {"query": "答对"}}],
        [{"name": "calc", "args": {"expression": "100 - 91.67"}}],
        "修复后准确率 100%，修复前 91.67%，提升 8.33 个百分点。",
    ])

    print(f"模型：{'qwen2.5:7b（真实）' if real else 'MockModel（离线）'}")
    print(f"问题：{QUESTION}\n")

    answer, steps = react_bindtools(QUESTION, model, verbose=True)

    print(f"\n答案：{answer}")
    print(f"步数：{steps}")
    return 0 if answer else 1


if __name__ == "__main__":
    sys.exit(main())
