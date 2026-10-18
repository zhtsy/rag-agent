import os
import datetime
import json
import re
import asyncio
import jieba
import gc
import torch
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from rich import print as rprint
from FlagEmbedding import FlagReranker
from bm25_retriever import bm25_retriever
from langchain_classic.retrievers import EnsembleRetriever
from graph_retriever import search_knowledge_graph

# ========== 1. 环境配置 ==========
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# ========== 2. 导入所需库 ==========
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document

# ========== 3. 初始化 RAG 组件 ==========
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_HOST
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 【修改点】强制重排模型在 CPU 上运行，把宝贵的 6GB 显存全留给 Qwen
device = 'cpu'
print(f"🔍 重排模型使用设备: {device}")
reranker = FlagReranker('/home/zh/models/bge-reranker-base', use_fp16=False)


# ========== 4. 定义工具 ==========
@tool
def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


def is_relation_query(query: str) -> bool:
    """用 DeepSeek 判断用户问题是否属于关系类查询（需要走图谱）"""
    prompt = f"""你是一个意图分类助手。请判断以下问题是否涉及"实体之间的关系"。
关系类查询包括：
- 谁住在哪个省/城市
- A和B是什么关系
- 某物是谁制造的
- 某人的宠物/家人/同事是什么
- 某物由什么材料制成
- 某人属于哪个组织/团队
- 某人在哪里工作
- 某人和某人有什么共同点

普通查询包括：
- 某人喜欢什么（属性查询，不是关系）
- 某物的特点是什么
- 某道菜怎么做
- 某游戏有什么玩法

请只回答"是"或"否"，不要解释。
问题：{query}"""

    try:
        response = cloud_llm.invoke([HumanMessage(content=prompt)])
        result = "是" in response.content.strip()
        print(f"🧠 DeepSeek 意图判断: {response.content.strip()} → {'走图谱' if result else '走向量'}")
        return result
    except Exception as e:
        print(f"⚠️ DeepSeek 不可用 ({e})，回退到关键词匹配")
        keywords = ["省", "属于", "关系", "制造者", "来源", "谁和谁", "共同点", "位于", "养", "工作", "由什么制成"]
        return any(kw in query for kw in keywords)


# 【修改点】增加 original_query 兜底参数，增加 try-except 捕获异常防止崩溃
@tool
def search_knowledge_base(query: str, original_query: str = "") -> str:
    """从本地知识库中检索与问题相关的信息，并进行重排序以提升相关性。"""
    try:
        text_to_check = original_query if original_query else query

        if any(kw in text_to_check for kw in ["为什么", "怎么", "如何"]):
            print(f"🔀 因果解释类问题，走向量检索")
        elif is_relation_query(text_to_check):
            print(f"🔀 关系查询，转给图谱检索")
            return search_knowledge_graph.invoke({"query": query, "original_query": text_to_check})

        known_entities = [
            "小智", "小明", "小王", "小赵", "小李", "小张", "小刘", "小陈", "小周", "小吴",
            "加里奥", "泰拉瑞亚", "川菜", "粤菜", "北京烤鸭", "东北美食", "江浙美食", "西北美食",
            "番茄炒蛋", "青椒炒蛋", "黄瓜炒鸡蛋", "韭菜炒鸡蛋", "木须肉", "洋葱炒蛋",
            "鸡蛋羹", "虎皮鸡蛋", "苦瓜炒蛋", "荷包蛋烧豆腐",
        ]
        query_entities = [e for e in known_entities if e in text_to_check]

        if query_entities:
            entity_filter = query_entities[0]
            vector_ret = vectorstore.as_retriever(search_kwargs={"k": 30, "filter": {"entity": entity_filter}})
            ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_ret], weights=[0.4, 0.6])
        else:
            entity_filter = None
            vector_ret = vectorstore.as_retriever(search_kwargs={"k": 10})
            ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_ret], weights=[0.4, 0.6])

        docs = multi_hop_retrieve(text_to_check, ensemble, max_hops=2)

        if query_entities:
            docs = [d for d in docs if d.metadata.get("entity") == entity_filter]
            print(f"🔍 实体过滤后剩余: {len(docs)} 个块")
            if not docs:
                return "未在知识库中找到相关信息。"

        if not docs:
            return "未在知识库中找到相关信息。"

        pairs = [[text_to_check, doc.page_content] for doc in docs]
        scores = reranker.compute_score(pairs, normalize=True)
        sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        stopwords = {"什么", "为什么", "怎么", "如何", "哪里", "哪个", "介绍", "告诉", "根据", "知识库", "的", "了",
                     "是", "能", "会", "被", "把", "和", "与"}
        core_keywords = [w for w in jieba.cut(text_to_check) if len(w) >= 2 and w not in stopwords]

        priority_docs = []
        other_docs = []
        for doc, score in sorted_docs:
            if any(kw in doc.page_content for kw in core_keywords):
                priority_docs.append((doc, score))
            else:
                other_docs.append((doc, score))

        top_docs = priority_docs[:4]
        for doc, score in other_docs:
            if len(top_docs) >= 5:
                break
            top_docs.append((doc, score))

        print(
            f"🔍 优先块 {len(priority_docs[:4])} 个 + 其他 {len(top_docs) - len(priority_docs[:4])} 个 = 共 {len(top_docs)} 个块")

        results = []
        for doc, score in top_docs:
            source = doc.metadata.get("source", "未知来源")
            results.append(f"【来源：{source}】\n{doc.page_content}")
        return "\n\n---\n\n".join(results)

    except Exception as e:
        import traceback
        print(f"❌ 检索工具内部发生致命错误:\n{traceback.format_exc()}")
        return f"检索工具执行失败，错误原因: {str(e)}"


def multi_hop_retrieve(query: str, retriever, max_hops=2):
    """规则驱动的多跳检索"""
    print(f"🔍 开始多跳检索，查询：{query}")

    def extract_keywords(text, min_len=2):
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', text)
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
                     '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        return [w for w in words if w not in stopwords and len(w) >= min_len]

    collected_docs = []
    original_keywords = extract_keywords(query)
    current_query = query
    last_query = None

    for hop in range(max_hops):
        if current_query == last_query:
            print("⚠️ 查询未变化，提前终止")
            break
        last_query = current_query
        print(f"🔁 第 {hop + 1} 轮检索，查询：{current_query}")

        docs = retriever.invoke(current_query)
        if not docs:
            break
        collected_docs.extend(docs)

        if len(collected_docs) >= 10:
            print(f"  已检索到 {len(collected_docs)} 个文档，停止检索")
            break

        if hop == 0 and len(docs) < 5:
            combined_text = " ".join([d.page_content for d in docs])
            covered = sum(1 for kw in original_keywords if kw in combined_text)
            if covered < len(original_keywords):
                missing_keywords = [kw for kw in original_keywords if kw not in combined_text]
                if missing_keywords:
                    current_query = " ".join(missing_keywords[:3])
                    print(f"  生成第二轮查询：{current_query}")
                    continue
        break

    seen = set()
    unique_docs = []
    for doc in collected_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    print(f"✅ 多跳检索完成，共 {len(unique_docs)} 个文档")
    return unique_docs


tools = [get_current_time, add, search_knowledge_base, search_knowledge_graph]
tool_node = ToolNode(tools)


# ========== 5. 结构化输出模型 ==========
class AgentAnswer(BaseModel):
    sources: list[str] = Field(description="回答所引用的知识库来源")
    confidence: float = Field(description="对回答的信心度，范围0-1")
    answer: str = Field(description="对用户问题的直接回答")


parser = PydanticOutputParser(pydantic_object=AgentAnswer)
format_instructions = parser.get_format_instructions()


# ========== 6. 状态定义 ==========
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ========== 7. 模型节点 ==========
load_dotenv()

# 【修改点】加上 num_ctx=2048 和 timeout=60，防止爆显存和死等
local_llm = ChatOllama(
    model="qwen2.5:7b",
    base_url=OLLAMA_HOST,
    temperature=0.0,
    num_predict=1024,
    num_ctx=2048,
    timeout=60
)

cloud_llm = ChatDeepSeek(model="deepseek-chat", temperature=0, timeout=60, max_retries=3)


def is_complex_query(query: str) -> bool:
    complex_keywords = ["为什么", "原因", "推理", "解释", "分析", "比较", "原理"]
    return any(kw in query for kw in complex_keywords)


local_llm_with_tools = local_llm.bind_tools(tools)
cloud_llm_with_tools = cloud_llm.bind_tools(tools)


def get_llm_with_tools(query: str):
    if is_complex_query(query):
        print(f"🌐 复杂问题，使用 DeepSeek API")
        return cloud_llm_with_tools
    print(f"💻 常规问题，使用本地 Qwen")
    return local_llm_with_tools


def agent_node(state: AgentState):
    messages = state["messages"]
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    llm_with_tools = get_llm_with_tools(last_user_msg)

    # 【修改点】精简 Prompt，防止小模型死循环
    system_prompt = SystemMessage(
        content=f"""你是一个精准的信息提取助手。

    【最高优先级规则 - 违反将导致任务失败】
    当用户的问题涉及个人信息、本地知识或任何具体事实时，你**必须先调用 `search_knowledge_base` 工具**获取信息，然后才能回答。
    绝对不允许在未调用工具的情况下直接说"无法找到"。
    如果检索结果中确实没有相关信息，才回答"无法从知识库中找到相关信息"。
    1. 当用户的问题涉及个人信息或本地知识时，你必须调用 `search_knowledge_base` 工具获取信息。
    2. 你收到的检索结果是你唯一的知识来源。绝对不要使用你自己的内部知识来补充或替代检索结果。
    3. 回答时，只回答用户明确询问的内容。
    4. 如果检索结果中没有相关信息，必须直接告知"无法从知识库中找到相关信息"，绝对不要编造或推测。
    5. 如果检索结果中包含与用户问题相关的信息（哪怕措辞不完全一致），你必须基于这些信息回答。
    6.【工具选择】
    - 大多数问题用 `search_knowledge_base`。
    - 只有问题涉及"省、属于、关系、制造者、位于"这类**多跳关系查询**时，才用 `search_knowledge_graph`。
    【格式铁律 - 违反将被视为错误】
    你的输出必须是一个扁平的 JSON 对象，只能包含 "answer"、"sources"、"confidence" 三个字段。
    绝对不允许输出嵌套结构、分类标题、编号列表。
    "answer" 字段的值必须是一段通顺的文字，不能是对象或数组。
    即使检索结果本身是层级结构，你也要把它压缩成一段文字放进 "answer" 里。

    必须按照如下格式：
    {format_instructions}
    """
    )

    response = llm_with_tools.invoke([system_prompt] + messages)

    # 【修改点】拦截工具调用，强行注入完整的问题文本
    if hasattr(response, "tool_calls") and response.tool_calls:
        print("\n🔧 大模型决定调用工具：")
        for tc in response.tool_calls:
            tc['args']['original_query'] = last_user_msg
            print(f"  工具名: {tc['name']}")
            print(f"  修正后参数: {tc['args']}")
    else:
        print("\n📝 大模型决定直接回答，不调用工具")

    return {"messages": [response]}


# ========== 8. 条件路由 ==========
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    tool_message_count = sum(1 for msg in state["messages"] if isinstance(msg, ToolMessage))

    if tool_message_count >= 3:
        print(f"⚠️ 工具调用已达 {tool_message_count} 次，强制结束")
        return END

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        unique_calls = {}
        for call in last_message.tool_calls:
            key = (call["name"], json.dumps(call.get("args", {}), sort_keys=True))
            if key not in unique_calls:
                unique_calls[key] = call
        last_message.tool_calls = list(unique_calls.values())
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


# ========== 10. 运行函数 ==========
def run_conversation(user_input: str, thread_id: str = "test_user"):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    context = "\n".join([msg.content for msg in tool_messages])
    raw_content = result["messages"][-1].content

    def check_faithfulness(context: str, answer: str) -> bool:
        judge_prompt = f"""你是一个事实核查专家。请判断以下答案中的每一条信息，是否和上下文有任何明显矛盾，如没有就回答"是"。
上下文：
{context}
答案：
{answer}
请只回答"是"或"否"。不要解释。"""
        response = local_llm.invoke([HumanMessage(content=judge_prompt)])
        return "是" in response.content

    try:
        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if json_match:
            structured_output = parser.parse(json_match.group(0))

            if structured_output.confidence < 0.5:
                return AgentAnswer(answer="抱歉，我无法从知识库中找到足够的信息来回答您的问题。", sources=[],
                                   confidence=0.0)

            used_graph = any("图谱检索结果" in msg.content for msg in result["messages"] if
                             hasattr(msg, "content") and isinstance(msg.content, str))

            if not used_graph and not check_faithfulness(context, structured_output.answer):
                return AgentAnswer(
                    answer=structured_output.answer + "\n\n⚠️ 以上回答部分内容可能未经上下文证实，请核实。",
                    sources=structured_output.sources, confidence=structured_output.confidence)
            return structured_output
        else:
            return raw_content
    except Exception as e:
        print(f"⚠️ 结构化解析失败: {e}")
        return raw_content