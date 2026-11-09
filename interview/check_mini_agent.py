#!/usr/bin/env python3
"""
mini_agent 自检：证明解析器 / 校验器 / 循环能扛住模型的典型乱输出。

刻意不叫 test_*.py —— 避免被项目根的 pytest 收集，污染既有 346 项基线。
直接跑：python3 check_mini_agent.py
"""
from mini_agent import MockLLM, ToolError, _clip, call_tool, run, validate

RESULTS: list[tuple[str, bool]] = []


def check(name: str, cond) -> None:
    RESULTS.append((name, bool(cond)))


def raises(fn, *args) -> bool:
    try:
        fn(*args)
        return False
    except ToolError:
        return True


# 1) 正常两跳：检索 -> 计算 -> 收敛
tr = run("Q", MockLLM([
    'Action: search_docs\nAction Input: {"query": "答对"}',
    'Action: calc\nAction Input: {"expression": "(48-44)/48*100"}',
    "Final Answer: 8.33 个百分点",
]))
check("正常两跳收敛", tr.stop_reason == "final" and len(tr.steps) == 3)

# 2) 参数名写错 -> 错误回灌 -> 模型自己改对
tr = run("Q", MockLLM([
    'Action: calc\nAction Input: {"expr": "1+1"}',
    'Action: calc\nAction Input: {"expression": "1+1"}',
    "Final Answer: 2",
]))
check("未知参数回灌后纠正",
      tr.answer == "2" and "未知参数" in tr.steps[0].observation)

# 3) Action Input 不是 JSON -> 回灌提示格式
tr = run("Q", MockLLM([
    "Action: calc\nAction Input: expression=1+1",
    'Action: calc\nAction Input: {"expression": "1+1"}',
    "Final Answer: 2",
]))
check("非法 JSON 回灌后纠正",
      tr.answer == "2" and "不是合法 JSON" in tr.steps[0].observation)

# 4) 编造工具名 -> 不能崩，错误回灌
tr = run("Q", MockLLM([
    'Action: web_search\nAction Input: {"q": "x"}',
    "Final Answer: 放弃",
]))
check("不存在的工具不崩", "不存在工具" in tr.steps[0].observation)

# 5) 原参数重复调用 -> 拦掉，避免死循环
tr = run("Q", MockLLM([
    'Action: calc\nAction Input: {"expression": "1+1"}',
    'Action: calc\nAction Input: {"expression": "1+1"}',
    "Final Answer: 2",
]))
check("重复调用被拦", "重复调用" in tr.steps[1].observation)

# 6) 模型完全不带格式 -> 回灌格式纠正
tr = run("Q", MockLLM(["我觉得答案是 8.33", "Final Answer: 8.33"]))
check("格式错误回灌后恢复",
      "格式错误" in tr.steps[0].observation and tr.answer == "8.33")

# 7) 步数耗尽 -> 明确返回 max_steps，不静默当成功
tr = run("Q", MockLLM(['Action: now\nAction Input: {"fmt": "date"}'] * 3), max_steps=3)
check("步数耗尽返回 max_steps",
      tr.stop_reason == "max_steps" and tr.answer is None and len(tr.steps) == 3)

# 8) 参数校验器
SCHEMA = {"required": ["a"], "properties": {"a": {"type": "integer"}}}
check("缺必填被拒", raises(validate, SCHEMA, {}))
check("类型不符被拒", raises(validate, SCHEMA, {"a": "1"}))
check("bool 冒充 integer 被拒", raises(validate, SCHEMA, {"a": True}))
check("多余参数被拒", raises(validate, SCHEMA, {"a": 1, "b": 2}))
check("enum 越界被拒",
      raises(validate, {"properties": {"f": {"type": "string", "enum": ["date"]}}}, {"f": "time"}))
check("合法参数放行", validate(SCHEMA, {"a": 1}) is None)

# 9) calc 不走 eval：注入与语法串都必须拒
check("拒绝代码注入", raises(call_tool, "calc", {"expression": "__import__('os').system('id')"}))
check("拒绝分号串表达式", raises(call_tool, "calc", {"expression": "1+1; 2"}))
check("拒绝字符串运算", raises(call_tool, "calc", {"expression": "'a'*3"}))
check("calc 正常求值", call_tool("calc", {"expression": "(48-44)/48*100"}) == "8.3333")

# 10) 自作主张写的 Observation 必须被截断
check("截断模型自写的 Observation",
      _clip('Action: x\nAction Input: {}\nObservation: 我猜是 1') == "Action: x\nAction Input: {}")

# 11) ```json 围栏能剥掉
tr = run("Q", MockLLM([
    'Action: calc\nAction Input: ```json\n{"expression": "2*3"}\n```',
    "Final Answer: 6",
]))
check("```json 围栏能剥掉", tr.answer == "6" and tr.steps[0].observation == "6")

# ---- 汇总 ----
bad = [n for n, ok in RESULTS if not ok]
for n, ok in RESULTS:
    print(("  ok   " if ok else "  FAIL ") + n)
print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} 通过")
raise SystemExit(1 if bad else 0)
