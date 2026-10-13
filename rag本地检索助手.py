import datetime
import os
from typing import TypedDict, Annotated, Sequence
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
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
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ========== 初始化向量数据库（指向刚才建好的库） ==========
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
# 创建检索器，每次取最相关的 3 个文档块
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


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

@tool
def search_knowledge_base(query: str) -> str:
    """【重要】当你需要回答关于用户个人信息、本地文档内容、或任何你不确定的问题时，必须调用此工具进行检索。不要自行猜测答案。"""
    docs = retriever.invoke(query)
    if not docs:
        return "未在知识库中找到相关信息，请告知用户无法回答。"
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

tools = [get_current_time, add,run_command,get_system_info,generate_random_number,calculate,search_knowledge_base]
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
    messages = state["messages"]
    system_prompt = SystemMessage(
        content="""你是一个精准的信息提取助手。
        你的回答必须严格基于以下原则：
        1. 当用户的问题涉及个人信息或本地知识时，你必须调用 `search_knowledge_base` 工具获取信息。
        2. 你收到的检索结果是你唯一的知识来源。绝对不要使用你自己的内部知识来补充或替代检索结果。
        3. 回答时，只回答用户明确询问的内容。例如，用户问“我的名字是什么”，你只回答名字，不要附带其他信息。
        4. 如果检索结果中没有相关信息，请直接告知“未找到相关信息”，不要编造或推测。
        """
    )
    # 将系统提示词放在消息列表的最前面
    response = llm_with_tools.invoke([system_prompt] + messages)
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

app = workflow.compile(checkpointer=checkpointer)

# ---------- 6. 测试运行 ----------
def run_conversation(user_input: str, thread_id: str = "test_user"):
    config = {"configurable": {"thread_id": thread_id}}
    # 注意：invoke 需要传入初始状态
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    # 提取最后一条消息（可能是 AIMessage 或 ToolMessage）
    last_msg = result["messages"][-1]
    # 如果是工具执行结果，直接输出 content
    if hasattr(last_msg, "content") and last_msg.content:
        return last_msg.content
    else:
        return "未获取到有效回答"

if __name__ == "__main__":
    # 测试多轮：先问时间，再问加法（同一会话应能记住之前的内容）
    # print("=== 第一轮：查询时间 ===")
    # print(run_conversation("现在几点了？"))
    #
    # print("\n=== 第二轮：计算加法（同一会话） ===")
    # print(run_conversation("23 加 45 等于多少？"))
    #
    # # 第三轮：询问记忆（测试是否记住上一轮）
    # print("\n=== 第三轮：引用记忆 ===")
    # print(run_conversation("刚才你算的加法结果是多少？"))
    #
    # print("\n=== 第四轮：查询当前系统运行信息 ===")
    # print(run_conversation("告诉我目前系统的运行信息是什么？"))
    #
    # print("\n=== 第五轮：生成一个随机整数，范围在0到100 ===")
    # print(run_conversation("告诉我生成的整数是什么？"))
    #
    # print("\n=== 第六轮：计算表达式 ===")
    # print(run_conversation("告诉我 100-99*11+100*22 等于多少？"))
    print("\n=== 第七轮：本地检索我的名字")
    print(run_conversation("根据知识库，我的名字叫什么？"))