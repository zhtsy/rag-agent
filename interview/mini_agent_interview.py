#!/usr/bin/env python3
"""
mini_agent_interview.py —— 面试白板版（不是工程版）

网上公认的「手撕标准」是 40~60 行，考察四块：
    工具注册 / 循环控制 / 终止条件 / 异常处理

这个文件：核心 react() 函数 42 行，其余是工具和演示用的假模型。
跑起来：python3 mini_agent_interview.py
"""
import ast
import json
import operator
import re

# ---------- 工具：一张 map，名字 -> 函数 ----------
DOCS = {"D1": "修复前答对 44/48（91.67%），修复后答对 48/48（100%）"}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
       ast.Mult: operator.mul, ast.Div: operator.truediv}


def _ev(n):
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in OPS:
        return OPS[type(n.op)](_ev(n.left), _ev(n.right))
    raise ValueError("只支持 + - * /")


def calc(expression):
    return round(_ev(ast.parse(expression, mode="eval").body), 4)


def search(query, top_k=1):
    hits = [v for k, v in DOCS.items() if any(w in v for w in query.split())]
    return "\n".join(hits) if hits else "没找到"


TOOLS = {"calc": calc, "search": search}

PROMPT = """你可以用工具。可用：calc(expression), search(query)

严格按格式输出，一次一个动作：
Thought: 想法
Action: 工具名
Action Input: 参数值

想清楚了就输出：
Final Answer: 答案

问题：{q}
"""


# ================= 核心：42 行 =================
def react(question, tools, llm, max_steps=6):
    msgs = [{"role": "user", "content": PROMPT.format(q=question)}]

    for _ in range(max_steps):
        raw = llm(msgs, stop=["\nObservation:"])
        msgs.append({"role": "assistant", "content": raw})

        m = re.search(r"Final Answer:\s*(.*)", raw, re.S)
        if m:
            return m.group(1).strip()

        name = re.search(r"Action:\s*(\w+)", raw)
        args = re.search(r"Action Input:\s*(.*)", raw, re.S)

        if not name:
            obs = "格式不对：要么 Action/Action Input，要么 Final Answer"
        elif name.group(1) not in tools:
            obs = f"没有工具 {name.group(1)}，可用：{list(tools)}"
        else:
            try:
                kw = json.loads(args.group(1).strip()) if args else {}
            except json.JSONDecodeError:
                obs = 'Action Input 要写成 {"参数名": "值"}'
            else:
                try:
                    obs = str(tools[name.group(1)](**kw))
                except Exception as e:
                    obs = f"工具报错：{e}"

        msgs.append({"role": "user", "content": f"Observation: {obs}"})

    return None
# ================= 核心结束 =================


class Mock:
    """假模型：按台词输出，不联网。"""

    def __init__(self, lines):
        self.lines, self.i = list(lines), 0

    def __call__(self, msgs, stop=None):
        out = self.lines[self.i] if self.i < len(self.lines) else "Final Answer: 完"
        self.i += 1
        return out


DEMO = [
    'Thought: 先查数据\nAction: search\nAction Input: {"query": "答对"}',
    'Thought: 差值是 100-91.67\nAction: calc\nAction Input: {"expression": "100-91.67"}',
    "Thought: 够了\nFinal Answer: 提升 8.33 个百分点",
]

if __name__ == "__main__":
    print(react("修复后比修复前提升几个百分点？", TOOLS, Mock(DEMO)))
