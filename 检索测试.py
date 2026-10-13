from rag_agent import retriever, vectorstore

# docs = retriever.invoke("小张")
# print("=== 查询：小张 ===")
# for d in docs[:5]:
#     print(d.metadata.get("source"), d.page_content[:80])
#
#
# results = vectorstore.similarity_search_with_score("小张", k=30)
# print("=== 查询：小张（带分数） ===")
# for doc, score in results:
#     source = doc.metadata.get("source")
#     print(f"分数: {score:.4f} | 来源: {source} | 内容: {doc.page_content[:60]}")

# docs = retriever.invoke("小吴 最近 学什么")
# print(f"检索到 {len(docs)} 个文档块：")
# for i, d in enumerate(docs[:10]):
#     print(f"\n--- 第 {i+1} 名 ---")
#     print(f"来源: {d.metadata.get('source')}")
#     print(f"人物: {d.metadata.get('person')}")
#     print(f"内容: {d.page_content[:200]}")
#     print(f"包含'潜水': {'潜水' in d.page_content}")

from rag_agent import vectorstore
docs = vectorstore.similarity_search_with_score("泰拉瑞亚 腐化 猩红 区别", k=10)
for doc, score in docs:
    print(f"距离: {score:.4f} | 包含'泰拉瑞亚': {'泰拉瑞亚' in doc.page_content}")
    print(f"内容: {doc.page_content[:100]}\n")