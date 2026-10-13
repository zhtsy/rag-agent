from rag_agent import vectorstore  # 直接复用你已初始化的向量库

def print_all_chunks():
    # 获取所有记录
    results = vectorstore.get()
    total = len(results["ids"])
    print(f"📦 向量库中共有 {total} 个文档块\n")

    for i in range(total):
        print(f"========== 块 {i+1} / {total} ==========")
        print(f"🆔 ID: {results['ids'][i]}")
        print(f"📁 来源: {results['metadatas'][i].get('source', '未知')}")
        print(f"👤 人物: {results['metadatas'][i].get('person', '无')}")
        print(f"📝 内容: {results['documents'][i][:300]}...")  # 只显示前300字符
        print()

if __name__ == "__main__":
    print_all_chunks()