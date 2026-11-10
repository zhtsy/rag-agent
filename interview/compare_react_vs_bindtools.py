#!/usr/bin/env python3
"""
compare_react_vs_bindtools.py —— 同一问题，两种协议，四组消融

面试时最常见的错觉是"用了 bind_tools 就比手写 ReAct 强"。这个脚本把它拆开：
影响结果的其实有两个独立变量，协议只是其中一个。

    变量一：工具描述写得好不好（模型靠它决定要不要调）
    变量二：工具实现对不对（这里指中文检索有没有正确切词）

四组实验把这两个变量分开：

    A  ReAct      + 弱描述 + split 切词    ← 全都不修
    B  ReAct      + 弱描述 + bigram 切词
    C  ReAct      + 强描述 + split 切词
    D  ReAct      + 强描述 + bigram 切词   ← 全都修好
    E  bind_tools + 强描述 + bigram 切词   ← 只换协议

对比 D 和 E，才能看出"协议本身"值多少。

跑法（需要 ollama + qwen2.5:7b）：
    python compare_react_vs_bindtools.py
"""
import time

from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

import mini_agent_interview as mi
from mini_agent_bindtools import react_bindtools, real_model, search as search_bigram

Q = "修复后的准确率比修复前提升几个百分点？"

WEAK_PROMPT = mi.PROMPT          # 原始白板版：只写了 calc(expression), search(query)
STRONG_PROMPT = """你可以用工具，规则：涉及项目具体数字先 search，涉及算数用 calc，不要让用户补充数据。

可用工具：
- calc(expression)：算数，expression 是纯算式，如 "100-91.67"
- search(query)：检索本项目资料库，里面存着评测准确率（修复前后各答对几条）、
  检索配置（文本块数、TOP_K、rerank 候选、阈值）、图谱结构。query 用短关键词。

严格按格式输出，一次一个动作：
Thought: 想法
Action: 工具名
Action Input: 参数名对应值的 JSON

想清楚了就输出：
Final Answer: 答案

问题：{q}
"""

BASE = ChatOllama(model="qwen2.5:7b", temperature=0)
_orig_search = mi.TOOLS["search"]      # 原始的 split 切词版，先存下来


def llm_react(msgs, stop=None):
    lc = [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
          for m in msgs]
    m = BASE.bind(stop=stop) if stop else BASE
    return m.invoke(lc).content


def run_react(prompt, search_fn):
    mi.PROMPT = prompt
    mi.TOOLS["search"] = search_fn
    try:
        return mi.react(Q, mi.TOOLS, llm_react)
    finally:
        mi.PROMPT = WEAK_PROMPT
        mi.TOOLS["search"] = _orig_search


def run_bind():
    ans, steps = react_bindtools(Q, real_model())
    return ans


def pick(ans):
    """只要答案里出现了 8.33 就算对，别的一律算没答上。"""
    if not ans:
        return "超步数"
    return "对" if "8.33" in ans else "错"


CASES = [
    ("A  ReAct      + 弱描述 + split", lambda: run_react(WEAK_PROMPT, _orig_search)),
    ("B  ReAct      + 弱描述 + bigram",
     lambda: run_react(WEAK_PROMPT, lambda query, top_k=1: search_bigram.invoke({"query": query}))),
    ("C  ReAct      + 强描述 + split", lambda: run_react(STRONG_PROMPT, _orig_search)),
    ("D  ReAct      + 强描述 + bigram",
     lambda: run_react(STRONG_PROMPT, lambda query, top_k=1: search_bigram.invoke({"query": query}))),
    ("E  bind_tools + 强描述 + bigram", run_bind),
]

if __name__ == "__main__":
    import sys
    repeat = 3
    if "--repeat" in sys.argv:
        repeat = int(sys.argv[sys.argv.index("--repeat") + 1])

    print(f"问题：{Q}")
    print(f"每组重复 {repeat} 次（单次结果不可靠，ReAct 版方差很大）\n")
    print(f"{'组别':34s}{'成功率':>8s}{'平均耗时':>10s}   失败样例")
    print("-" * 78)
    for name, fn in CASES:
        ok, total, sample = 0, 0.0, ""
        for _ in range(repeat):
            t = time.time()
            try:
                ans = fn()
            except Exception as e:                  # 单组崩了不影响其它组
                ans = f"<异常 {type(e).__name__}>"
            total += time.time() - t
            if pick(ans) == "对":
                ok += 1
            elif not sample:
                sample = str(ans)[:30]
        print(f"{name:34s}{f'{ok}/{repeat}':>8s}{total/repeat:9.1f}s   {sample!r}")
