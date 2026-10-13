from rag_agent import vectorstore

# 带过滤的检索器：只搜小吴
filtered = vectorstore.as_retriever(
    search_kwargs={"k": 30, "filter": {"person": "小吴"}}
)

docs = filtered.invoke("小吴 最近 学什么")
print(f"检索到 {len(docs)} 个块：")
for i, d in enumerate(docs):
    print(f"\n--- 第 {i+1} 名 ---")
    print(f"来源: {d.metadata.get('source')}")
    print(f"人物: {d.metadata.get('person')}")
    print(f"内容: {d.page_content[:150]}")