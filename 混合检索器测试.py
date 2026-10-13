# from langchain_classic.retrievers import EnsembleRetriever
# from bm25_retriever import bm25_retriever
# from rag_agent import vectorstore
#
# # 带实体过滤的向量检索器
# vector_ret_filtered = vectorstore.as_retriever(
#     search_kwargs={"k": 30, "filter": {"entity": "小陈"}}
# )
#
# # BM25 不支持过滤，但可以在融合后手动过滤
# ensemble = EnsembleRetriever(
#     retrievers=[bm25_retriever, vector_ret_filtered],
#     weights=[0.4, 0.6]
# )
#
# docs = ensemble.invoke("小陈 宠物")
# print(f"融合检索到 {len(docs)} 个块")
# for d in docs[:5]:
#     print(f"来源: {d.metadata.get('source')} | 内容: {d.page_content[:80]}")

from rag_agent import search_knowledge_base

# 测试多跳+混合
result = search_knowledge_base.invoke("小陈养了什么宠物？")
print(result[:500])

result = search_knowledge_base.invoke("虎皮鸡蛋怎么做？")
print(result[:500])