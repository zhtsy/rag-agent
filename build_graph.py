import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

OLLAMA_HOST = "http://127.0.0.1:11434"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

# 1. 加载所有文档块
embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
data = vectorstore.get()
chunks = [
    Document(page_content=data["documents"][i], metadata=data["metadatas"][i])
    for i in range(len(data["documents"]))
]

# 2. 初始化 LLM 和 GraphTransformer
load_dotenv()  # 加载 .env 里的 API Key

# 图谱抽取用云端 API
llm = ChatDeepSeek(
    model="deepseek-chat",       # DeepSeek 官方模型名
    temperature=0
)


# 定义允许的节点类型和关系类型（约束抽取范围，避免图谱太乱）
graph_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["人物", "地点", "游戏", "食物", "宠物", "职业", "菜谱", "组织"],
    allowed_relationships=[
        "喜欢", "不喜欢", "住在", "养了", "擅长", "属于",
        "制作", "由...制成", "拥有", "工作于"
    ]
)

# 3. 逐块抽取（小批量，避免本地模型超时）
graph_documents = []
for i, chunk in enumerate(chunks):
    print(f"抽取块 {i+1}/{len(chunks)}...")
    try:
        docs = graph_transformer.convert_to_graph_documents([chunk])
        graph_documents.extend(docs)
    except Exception as e:
        print(f"  块 {i+1} 抽取失败: {e}")

print(f"共抽取 {len(graph_documents)} 个图谱文档")



# ========== 存入 Neo4j ==========
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

print("开始写入 Neo4j...")
try:
    graph.add_graph_documents(graph_documents, baseEntityLabel=True)
    print("✅ 写入完成")
except Exception as e:
    print(f"❌ 写入失败: {e}")
    import traceback
    traceback.print_exc()

# ========== 立即验证 ==========
graph.refresh_schema()
result = graph.query("MATCH (n) RETURN count(n) AS node_count")
print(f"📊 当前节点数: {result}")
result2 = graph.query("MATCH ()-[r]->() RETURN count(r) AS rel_count")
print(f"📊 当前关系数: {result2}")
print(f"📊 Schema: {graph.schema}")