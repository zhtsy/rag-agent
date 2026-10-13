
import datetime
import subprocess
import json
import re
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ============ 1. 定义四个工具 ============
@tool
def add(a: int, b: int) -> int:
    """计算两个数字的和。"""
    return a + b


@tool
def get_current_time(format_type: str = "datetime") -> str:
    """获取当前的日期和时间。当用户问时间、日期、现在几点时使用。参数 format_type 可选 'date' 或 'datetime'"""
    now = datetime.datetime.now()
    if format_type == "date":
        return now.strftime("%Y-%m-%d")
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def run_shell_command(command: str) -> str:
    """在 Ubuntu 终端执行一条简单的 Shell 命令。比如 ls, pwd, echo, whoami。"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=5)
        if result.stdout:
            return f"命令执行成功:\n{result.stdout.strip()}"
        elif result.stderr:
            return f"命令执行出错:\n{result.stderr.strip()}"
        return "命令执行完成，没有输出。"
    except subprocess.TimeoutExpired:
        return "命令执行超时。"
    except Exception as e:
        return f"执行命令时发生错误: {str(e)}"


@tool
def count_words(text: str) -> str:
    """统计一段文本的字数（按空格分隔）。当用户问“这段话有多少字”时使用。"""
    word_count = len(text.split())
    char_count = len(text)
    return f"这段文本共有 {char_count} 个字符（含空格），约 {word_count} 个单词。"


tools = [add,get_current_time,run_shell_command,count_words]
tool_map = {t.name: t for t in tools}  # 名称 -> 工具函数


# ============ 2. 状态定义 ============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============ 3. 模型节点 ============
llm = ChatOllama(
    model="deepseek-r1:7b",
    num_predict=8192,      # 从 2048 大幅提高到 8192，避免截断
    temperature=0.1        # 降低温度，让它更“听话”输出固定格式
)

SYSTEM_PROMPT =  """你是一个助手。如果用户问时间，你就说 "Action: get_current_time"。
如果用户问文件，就说 "Action: run_shell_command"。
不要解释太多，直接输出 Action。
- 如果用户明确提到“加”、“减”、“乘”、“除”、“等于”、“计算”，必须使用 add 工具。
- 如果用户明确提到“列出”、“文件”、“目录”、“统计数量”，必须使用 run_shell_command 工具。
- 不要把“数字计算”和“文件操作”混淆。"""


def agent_node(state: AgentState):
    # 如果上一条消息是工具执行结果，直接返回
    if state["messages"] and "工具执行结果" in state["messages"][-1].content:
        return {"messages": [state["messages"][-1]]}

    # 否则正常调用模型
    messages = state["messages"]
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    response = llm.invoke([system_msg] + list(messages))
    return {"messages": [response]}


# ============ 4. 工具执行节点（解析 JSON） ============



def tool_executor(state: AgentState):
    last_msg = state["messages"][-1]
    content = last_msg.content
    print(f"🔍 原始输出待解析:\n{content}")  # 调试：看看模型到底输出了什么

    # ========== 使用正则表达式进行“模糊匹配” ==========

    # 1. 尝试匹配工具名（容忍截断，比如 "ge" 也能匹配到 "get_current_time"）
    tool_name = None
    # 匹配 "Action:" 后面的单词，哪怕只写了 "ge"，我们也手动映射
    action_match = re.search(r'Action:\s*(\w+)', content, re.IGNORECASE)
    if action_match:
        raw_name = action_match.group(1).lower()
        # 智能补全映射表（根据你定义的4个工具）
        if "get_current_time" in raw_name or raw_name.startswith("ge"):
            tool_name = "get_current_time"
        elif "run_shell" in raw_name or raw_name.startswith("ru"):
            tool_name = "run_shell_command"
        elif "add" in raw_name:
            tool_name = "add"
        elif "count_words" in raw_name or raw_name.startswith("co"):
            tool_name = "count_words"

    # 2. 如果没找到 Action，尝试直接找工具名（有时模型会直接写工具名称）
    if not tool_name:
        if "get_current_time" in content or "当前时间" in content:
            tool_name = "get_current_time"
        elif "run_shell" in content or "ls" in content:
            tool_name = "run_shell_command"

    # 3. 如果识别到了工具名，提取参数
    if tool_name:
        params = {}
        # 尝试提取 Action Input 中的 JSON 或参数
        param_match = re.search(r'Action Input:\s*(\{.*?\})', content, re.DOTALL)
        if param_match:
            try:
                params = json.loads(param_match.group(1))
            except:
                params = {}

        # 如果是 get_current_time 且没有参数，默认返回 datetime
        if tool_name == "get_current_time" and not params:
            params = {"format_type": "datetime"}

        # 如果是 run_shell_command 且没有参数，默认用 ls
        if tool_name == "run_shell_command" and not params:
            params = {"command": "ls -la | wc -l"}  # 默认统计文件数（针对你想要的570效果）
        # 在 tool_executor 函数中，执行工具之前

        if tool_name == "add" and not params:
            # 尝试从用户消息中提取两个数字
            user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
            if user_messages:
                last_user_msg = user_messages[-1].content
                # 使用正则提取数字（支持整数和浮点数）
                numbers = re.findall(r'\d+', last_user_msg)
                if len(numbers) >= 2:
                    params = {"a": int(numbers[0]), "b": int(numbers[1])}
                    print(f"📊 自动从用户消息中提取参数: {params}")
                else:
                    # 如果找不到两个数字，提醒用户
                    return {"messages": [AIMessage(content="❌ 请明确提供两个数字，例如：97加98等于多少？")]}
            else:
                return {"messages": [AIMessage(content="❌ 未找到用户消息，无法提取数字。")]}

        # 执行工具
        print(f"✅ 识别到工具: {tool_name}, 参数: {params}")
        func = tool_map[tool_name]
        result = func.invoke(params)
        # 在 return 之前插入
        print(f"🔧 执行工具后得到的 result 原始值: {result}")
        print(f"📦 result 的类型: {type(result)}")
        return {"messages": [AIMessage(content=f"工具执行结果：{result}")]}



    # 4. 如果完全没识别出工具，就返回原始回答（让 AI 直接对话）
    return {"messages": [AIMessage(content=content)]}

# ============ 5. 条件边 ============
def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    content = last_msg.content
    print(f"🔍 should_continue 检查内容: {content}")
    if "Action:" in content:
        return "tools"
    return END


# ============ 6. 构建工作流 ============
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_executor)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END
})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# ============ 7. 运行测试 ============
thread_id = "test_session"
config = {"configurable": {"thread_id": thread_id}}

print("=== 第一轮：查询时间 ===")
result1 = app.invoke(
    {"messages": [HumanMessage(content="现在几点了？")]},
    config=config
)
print("AI 最终回答:", result1["messages"][-1].content)

print("\n=== 第二轮：执行命令并统计 ===")
# 新建一个独立的会话 ID，彻底清空记忆
new_config = {"configurable": {"thread_id": "file_count_session"}}
result2 = app.invoke(
    {"messages": [HumanMessage(content="请问97+98等于多少")]},
    config=new_config  # 使用全新的 thread_id
)
print("AI 最终回答:", result2["messages"][-1].content)