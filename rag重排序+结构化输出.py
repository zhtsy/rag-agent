import os
import datetime
import json
import re
from typing import TypedDict, Annotated, Sequence

# ========== 1. 环境配置（消除兼容性警告） ==========
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 显式使用 GPU 0
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像加速

# ========== 2. 导入所需库 ==========
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import torch


# ========== 3. 定义工具 ==========
@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


# ========== 4. 初始化 RAG 组件 ==========
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # 多取一些用于重排序

# 初始化重排序模型（使用 GPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔍 使用设备: {device}")
reranker = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-6-v2',
    device = device
)


@tool
def search_knowledge_base(query: str) -> str:
    """从本地知识库中检索与问题相关的信息，并进行重排序以提升相关性。"""
    # 1. 初次检索
    docs = retriever.invoke(query)
    if not docs:
        return "未在知识库中找到相关信息。"

    # 2. 准备重排序输入
    pairs = [[query, doc.page_content] for doc in docs]

    # 3. 重排序（GPU 加速）
    scores = reranker.predict(pairs)

    # 4. 按分数排序，取前3个
    sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in sorted_docs[:3]]

    # 5. 返回内容和来源
    results = []
    for doc in top_docs:
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content
        results.append(f"【来源：{source}】\n{content}")
    return "\n\n---\n\n".join(results)


# 工具列表
tools = [get_current_time, add, search_knowledge_base]
tool_node = ToolNode(tools)


# ========== 5. 定义结构化输出模型 ==========
class AgentAnswer(BaseModel):
    answer: str = Field(description="对用户问题的直接回答")
    sources: list[str] = Field(description="回答所引用的知识库来源（文件名或章节），如果没有引用则为空列表")
    confidence: float = Field(description="对回答的信心度，范围0-1")


parser = PydanticOutputParser(pydantic_object=AgentAnswer)
format_instructions = parser.get_format_instructions()


# ========== 6. 状态定义 ==========
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ========== 7. 模型节点 ==========
llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://127.0.0.1:11434",
    temperature=0.0
)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    messages = state["messages"]
    system_prompt = SystemMessage(
        content=f"""你是一个精准的信息提取助手。
        你的回答必须严格基于以下原则：
        1. 当用户的问题涉及个人信息或本地知识时，你必须调用 `search_knowledge_base` 工具获取信息。
        2. 你收到的检索结果是你唯一的知识来源。绝对不要使用你自己的内部知识来补充或替代检索结果。
        3. 回答时，只回答用户明确询问的内容。
        4. 如果检索结果中没有相关信息，请直接告知"未找到相关信息"，不要编造或推测。
        5. 你的最终回答必须严格遵循以下 JSON 格式：
        {format_instructions}
        """
    )
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}


# ========== 8. 条件路由 ==========
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ========== 9. 构建工作流 ==========
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END
})
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# ========== 10. 运行函数（带结构化解析） ==========
def run_conversation(user_input: str, thread_id: str = "test_user"):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    raw_content = result["messages"][-1].content

    # 尝试提取 JSON 并结构化解析
    try:
        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            structured_output = parser.parse(json_str)
            return structured_output
        else:
            return raw_content
    except Exception as e:
        print(f"⚠️ 结构化解析失败: {e}")
        return raw_content


# ========== 11. 测试 ==========
if __name__ == "__main__":
    print("=== 测试1：本地知识库检索（含重排序） ===")
    result = run_conversation("根据知识库，我叫小赵，我喜欢做什么？")

    if hasattr(result, "answer"):
        print(f"答案: {result.answer}")
        print(f"来源: {result.sources}")
        print(f"置信度: {result.confidence}")
    else:
        print(result)

    print("=== 测试2：本地知识库检索（含重排序） ===")
    result = run_conversation("根据知识库，我叫小明，我喜欢干什么？我不喜欢什么？")

    if hasattr(result, "answer"):
        print(f"答案: {result.answer}")
        print(f"来源: {result.sources}")
        print(f"置信度: {result.confidence}")
    else:
        print(result)

    print("=== 测试3：本地知识库检索（含重排序） ===")
    result = run_conversation("根据知识库，我叫小明，我的银行卡密码是多少？")

    if hasattr(result, "answer"):
        print(f"答案: {result.answer}")
        print(f"来源: {result.sources}")
        print(f"置信度: {result.confidence}")
    else:
        print(result)