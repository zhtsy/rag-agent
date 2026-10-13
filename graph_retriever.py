import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool

load_dotenv()

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

# ========== Cypher 生成提示词 ==========
CYPHER_GENERATION_TEMPLATE = """任务：根据用户的问题，生成一个 Neo4j Cypher 查询语句。
你只能使用以下 Schema 中提供的节点、关系和属性：
{schema}



规则：
1. 只输出 Cypher 语句本身，不要生成任何解释。
2. 如果问题涉及"省份"、"省"，必须使用两跳查询：人物 -[:住在]-> 城市 -[:属于]-> 省份。
3. 【重要】查询结果必须返回节点的**所有属性**（用 properties() 函数），而不仅仅是 id。
   例如：RETURN properties(pet) 而不是 RETURN pet.id


示例1（单跳，返回属性）：
用户问：小陈住在哪里？
Cypher: MATCH (p {{id: "小陈"}})-[:住在]->(city) RETURN properties(city)

示例2（多跳，返回属性）：
用户问：小陈住在哪个省？
Cypher: MATCH (p {{id: "小陈"}})-[:住在]->(city)-[:属于]->(province) RETURN properties(province)

示例3（宠物查询，返回属性）：
用户问：小陈养了什么宠物？
Cypher: MATCH (p {{id: "小陈"}})-[:养了]->(pet) RETURN properties(pet)

现在，请为以下问题生成 Cypher：
用户问：{question}
Cypher:"""

cypher_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"]
)

# ========== QA 回答提示词（不是 Cypher 模板！） ==========
QA_TEMPLATE = """根据以下查询结果，直接回答用户的问题。不要添加任何查询结果之外的信息。

查询结果：
{context}

用户问题：{question}
回答："""

qa_prompt = PromptTemplate(
    template=QA_TEMPLATE,
    input_variables=["context", "question"]
)

# ========== 创建 Chain ==========
cypher_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    return_intermediate_steps=True,
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt
)


@tool
def search_knowledge_graph(query: str) -> str:
    """从知识图谱中检索实体和关系信息。当用户询问需要多跳推理的问题时使用，
    例如'小陈住在哪个省'、'加里奥的制造者是谁'、'谁和谁有什么关系'。"""
    try:
        result = cypher_chain.invoke({"query": query})
        if result and result.get("result"):
            return f"【图谱检索结果】\n{result['result']}"
        return "图谱中未找到相关信息。"
    except Exception as e:
        return f"图谱检索出错: {str(e)}"