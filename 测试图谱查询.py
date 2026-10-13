from graph_retriever import cypher_chain

# 测试单跳查询
print("=== 小陈住在哪里？===")
print(cypher_chain.invoke({"query": "小陈住在哪里？"}))

# 测试多跳查询（图谱的核心优势）
print("\n=== 小陈住在哪个省？===")
print(cypher_chain.invoke({"query": "小陈住在哪个省？"}))