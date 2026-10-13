from rag_agent import retriever

# 测试 T007
docs = retriever.invoke("小张 甜点")
print("=== 小张 甜点 检索结果 ===")
for i, d in enumerate(docs[:10]):
    print(f"{i+1}. 来源: {d.metadata.get('source')} | 内容: {d.page_content[:80]}")

# 测试 T009
docs = retriever.invoke("小陈 宠物")
print("\n=== 小陈 宠物 检索结果 ===")
for i, d in enumerate(docs[:10]):
    print(f"{i+1}. 来源: {d.metadata.get('source')} | 内容: {d.page_content[:80]}")

# 测试 T013
docs = retriever.invoke("川菜 味型 数量")
print("\n=== 川菜 味型 检索结果 ===")
for i, d in enumerate(docs[:10]):
    print(f"{i+1}. 来源: {d.metadata.get('source')} | 内容: {d.page_content[:80]}")