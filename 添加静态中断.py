import datetime
import os
from typing import TypedDict, Annotated, Sequence
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
import os
import json
import requests
import subprocess
import random
import time


# ---------- 1. 定义工具 ----------
@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


@tool
def run_command(command: str) -> str:
    """在 WSL/Ubuntu 终端执行一条 Shell 命令。适用于系统管理、文件操作、程序运行等。注意安全！"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        error = result.stderr.strip()
        if output and error:
            return f"📤 输出:\n{output}\n\n⚠️ 错误信息:\n{error}"
        elif output:
            return f"📤 执行成功:\n{output}"
        elif error:
            return f"⚠️ 执行出错:\n{error}"
        return "✅ 命令执行完成，无输出"
    except subprocess.TimeoutExpired:
        return "⏱️ 命令超时（30秒），已终止"
    except Exception as e:
        return f"❌ 执行失败: {str(e)}"


@tool
def get_system_info() -> str:
    """获取当前系统的运行状态信息，包括 CPU、内存、磁盘、进程等。"""
    try:
        # CPU 信息
        cpu = subprocess.run("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1", shell=True,
                             capture_output=True, text=True)
        # 内存信息
        mem = subprocess.run("free -h | grep Mem | awk '{print $3\"/\"$2}'", shell=True, capture_output=True, text=True)
        # 磁盘信息
        disk = subprocess.run("df -h / | tail -1 | awk '{print $3\"/\"$2 \" (\" $5 \")\"}'", shell=True,
                              capture_output=True, text=True)
        # 进程数
        procs = subprocess.run("ps aux | wc -l", shell=True, capture_output=True, text=True)

        return f"""📊 系统状态:
- CPU 使用率: {cpu.stdout.strip()}%
- 内存使用: {mem.stdout.strip()}
- 磁盘使用: {disk.stdout.strip()}
- 运行进程: {procs.stdout.strip()} 个
- 当前用户: {subprocess.run('whoami', shell=True, capture_output=True, text=True).stdout.strip()}"""
    except Exception as e:
        return f"获取系统信息失败: {str(e)}"

@tool
def generate_random_number(min_val: int = 1, max_val: int = 100) -> str:
    """生成一个指定范围内的随机整数。适用于抽奖、随机采样等场景。"""
    num = random.randint(min_val, max_val)
    return f"🎲 随机数 ({min_val} ~ {max_val}): {num}"

@tool
def calculate(expression: str) -> str:
    """安全地计算一个数学表达式的值。支持 +, -, *, /, **, sqrt() 等基本运算。不执行任何系统命令。"""
    import math
    # 只允许数字、运算符、括号和常用数学函数
    allowed = set("0123456789+-*/()., %sqrtlogexp sin cos tan pi e")
    for char in expression:
        if char not in allowed and not char.isalpha():
            return f"❌ 表达式包含非法字符: {char}"
    try:
        # 使用 eval 但限制在 math 模块内
        result = eval(expression, {"__builtins__": {}}, {"math": math, "pi": math.pi, "e": math.e, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "exp": math.exp})
        return f"🧮 {expression} = {result}"
    except Exception as e:
        return f"❌ 计算错误: {str(e)}"

tools = [get_current_time, add,run_command,get_system_info,generate_random_number,calculate]
tool_node = ToolNode(tools)  # 自动执行工具



checkpointer = MemorySaver()

# ---------- 2. 状态定义 ----------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ---------- 3. 模型节点 ----------
llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://127.0.0.1:11434",   # 明确指定地址
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools)

def human_approval_node(state: AgentState):
    """需要人工审批的节点"""
    messages = state["messages"]
    last_message = messages[-1]

    # 1. 检查是否需要人工审批
    # 这里模拟检查，你可以根据自己的逻辑进行判断
    # 例如，检查消息中是否包含 "删除" 等敏感词
    if "删除" in last_message.content:
        # 2. 需要审批：调用 interrupt() 暂停执行
        # 将需要人类审查的信息传入 interrupt 函数[reference:12]
        approval_request = {
            "message": "即将执行敏感操作，请审批",
            "details": last_message.content,
            "action": "confirm_deletion"
        }
        # 执行到这里会暂停，等待外部输入[reference:13][reference:14]
        user_input = interrupt(approval_request)

        # 3. 处理用户返回的审批结果
        # interrupt() 的返回值就是用户通过 Command(resume=...) 提供的数据[reference:15]
        if user_input and user_input.get("approved") == "yes":
            # 用户批准，继续执行
            return {"messages": messages + [AIMessage(content="操作已获批准。")]}
        else:
            # 用户拒绝，阻止后续操作
            return {"messages": messages + [AIMessage(content="操作已被用户取消。")]}

    # 不需要审批，直接返回
    return {"messages": messages}
def agent_node(state: AgentState):
    """调用模型，并返回包含 tool_calls 的消息"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    # 打印调试信息（可选）
    # print(f"模型返回: {response}")
    # print(f"工具调用: {response.tool_calls if hasattr(response, 'tool_calls') else '无'}")
    return {"messages": [response]}

# ---------- 4. 条件路由 ----------
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # 如果模型返回了 tool_calls，则去工具节点
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# ---------- 5. 构建工作流 ----------
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
# workflow.add_node("human_approval", human_approval_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END
})
# workflow.add_edge("human_approval", "tools")
workflow.add_edge("tools", "agent")   # 工具执行后回到 agent

# 添加记忆（支持多轮）

app = workflow.compile(checkpointer=checkpointer,interrupt_before=["tools"])

# ---------- 6. 测试运行 ----------
def run_conversation(user_input: str, thread_id: str = "test_user"):
    # 原来的 run_conversation 可以先注释掉，我们用这个新逻辑测试

    config = {"configurable": {"thread_id": "test_user"}}

    # 1. 第一次执行：用户提问，Agent 思考后会卡在 tools 节点前
    for event in app.stream(
            {"messages": [HumanMessage(content="现在几点了？")]},
            config=config,
            stream_mode="values"
    ):
        # 打印中间状态（方便你看卡在哪了）
        print(event["messages"][-1].content if event.get("messages") else event)

    # 2. 检查状态：确实卡住了吗？
    state = app.get_state(config)
    print(f"\n⏸️ 当前是否卡住？{state.next}")  # 如果输出 ('tools',)，说明确实卡住了

    # 3. 模拟人工确认（现实中这里是前端按钮或命令行输入）
    user_decision = input("\n是否允许执行工具？(输入 yes 继续): ")

    # 4. 恢复执行
    for event in app.stream(
            Command(resume=user_decision),  # 把用户的决定传进去
            config=config,
            stream_mode="values"
    ):
        # 最终结果会在这里打印出来
        if event.get("messages"):
            print("🤖 最终回答:", event["messages"][-1].content)

if __name__ == "__main__":
    # 测试多轮：先问时间，再问加法（同一会话应能记住之前的内容）
    print("=== 第一轮：查询时间 ===")
    print(run_conversation("现在几点了？"))

  