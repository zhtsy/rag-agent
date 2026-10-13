import os
import datetime
import json
import re
from typing import TypedDict, Annotated, Sequence



# ========== 1. 环境配置（消除兼容性警告） ==========
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 显式使用 GPU 0
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
from collections import Counter

# ========== 3. 定义工具 ==========

@tool
def get_current_time():
    """获取当前时间"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def add(a,b):
    """获取两个数相加的结果"""
    return a + b

def search_knowledge_base(query):
    """从本地知识库中检索与问题相关的信息，并进行重排序以提升相关性。"""
    #多跳检索（规则驱动）
    docs = multi_hop_retrieve(query, retriever, max_hops=2)
    if not docs:
        return "未在知识库中找到相关信息"

    #重排序
    pairs = [[query, doc.page_content]for doc in docs]
    scores = reranker.predict(pairs)
    sorted_docs = sorted(zip(docs,scores), key=lambda x: x[1], reverse=True)
    top_docs = sorted_docs[:5]

    results = []
    for doc,score in top_docs:
        source = doc.metadata.get("source","未知来源")#提取数据来源
        content = doc.page_content #提取文档正文内容
        results.append(f"【来源：{source} | 相关度：{score:.3f}】\n{content}")
    return "\n\n---\n\n".join(results)
def multi_hop_retrieve(query, retrieve, max_hops=2):
    """
    规则驱动的多跳检索（不依赖 LLM 判断）：
    1. 第一轮：用原始问题检索
    2. 如果检索到的文档少于 5 个，或者关键词覆盖不全，则：
       - 提取问题中的关键词（取前 3 个有意义的词）
       - 用这些关键词组合成新的查询，进行第二轮检索
    3. 合并两轮结果并去重

    """
    def extract_keywords(text , min_len=2):
        words =re.findall(r'[\u4e00-\u9fa5a-zA-Z]+',text)
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
                     '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        keywords = [w for w in words if w not in stopwords and len(w) >= min_len]
        return keywords

    collected_docs = []
    original_keywords = extract_keywords(query)
    current_query = query

    for hop in range(max_hops):
        print(f"🔁 第{hop+1}轮检索，查询：{current_query}")
        #检索
        docs = retriever.invoke(current_query)
        if not docs:
            break
        collected_docs.extend(docs)
        if len(collected_docs) >= 10:
            print(f"已检索到{len(collected_docs)}个文档，停止检索。")
            break

        if hop == 0 and len(docs) < 5:
            #检查当前结果是否覆盖了原始关键词
            combined_text = " ".join([d.page_content for d in docs])
            covered = sum(1 for kw in original_keywords if kw in combined_text)
            print(f"关键覆盖率：{covered} / {len(original_keywords)}")

            if covered < len(original_keywords):
                missing_keywords = [kw for kw in original_keywords if kw not in combined_text]
                if missing_keywords:
                    current_query = " ".join(missing_keywords)
                    print(f"生成第二次查询：{current_query}")
                    continue

        # 默认：用原始查询，但只做一轮（防止无限循环）
        break
    #去重 （按内容去重）
    seen = set()
    unique_docs = []
    for doc in collected_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    print(f"✅ 多跳检索完成，共 {len(unique_docs)} 个文档")
    return unique_docs

#工具列表
tools = [get_current_time,add,search_knowledge_base]
tool_node = ToolNode(tools)

# ========== 4. 初始化 RAG 组件 ==========
embedding_model = OllamaEmbeddings(
    model = "nomic-embed-text",
    base_url = "http://127.0.0.1:11434"

)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model

)
retriever = vectorstore.as_retriever(search_kwargs={"k": 30})#多取一些用于重排序

#初始化重排序模型（使用 GPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔍 使用设备: {device}")
reranker = CrossEncoder(
    '/home/zh/models/ms-marco-MiniLM-L-6-v2',
    device=device

)

# ========== 5. 定义结构化输出模型 ==========
class AgentAnswer(BaseModel):
    answer: str = Field(description="对用户问题的直接回答")
    sources: list[str] = Field(description="回答所引用的知识库来源（文件名或章节），如果没有引用则为空列表")
    confidence: float = Field(description="对回答的信心度，范围0-1")

parser = PydanticOutputParser(pydantic_object=AgentAnswer)
# 在 LangChain 中，JsonOutputParser 默认返回的是 Python 字典（dict），而不是对象。
# 只有当你显式定义了 Pydantic BaseModel 并用 PydanticOutputParser 时，才会返回对象实例。
# 因为下文的 if hasattr（）只能用于对象
format_instructions = parser.get_format_instructions()

# ========== 6. 状态定义 ==========
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]


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
        6. 请根据知识库内容，回答用户的问题。如果存在多个相关信息，请在答案中全部列出，并引用对应的来源文件。
        7. “你是一个严谨的问答助理。对于用户的问题，请先判断文档中的信息是否在‘正向回答’该问题。  
        例如，用户问‘喜欢什么’，文档中的‘讨厌’就不属于该问题的有效回答。  
        请在最终答案中只包含**正向匹配**的信息。”
        
        """
    )
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}

# ========== 8. 条件路由 ==========
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ========== 9. 构建工作流 ==========
workflow = StateGraph(AgentState)
workflow.add_node("agent",agent_node)
workflow.add_node("tools",tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent",should_continue,{
    "tools": "tools",
    END: END
})
workflow.add_edge("tools","agent")

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ========== 10. 运行函数（带结构化解析） ==========
def run_conversation(user_input: str,thread_id: str ="test_user"):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    raw_content = result["messages"][-1].content

    # 尝试提取JSON 并结构化解析
    try:
        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            structured_output = parser.parse(json_str)
            return structured_output
        else:
            return raw_content
    except Exception as e:
        print(f"结构化失败:{e}")
        return raw_content

# ========== 11. 测试 ==========
if __name__ == "__main__":
    print("=== 测试1：本地知识检索（含重排序）===")
    result = run_conversation("根据知识库，我的名字叫小智，我喜欢做什么？")

    if hasattr(result, "answer"):
        print(f"答案：{result.answer}")
        print(f"来源：{result.sources}")
        print(f"置信度：{result.confidence}")

    else:
        print(result)






