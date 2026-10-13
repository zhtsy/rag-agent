import time
import requests

def test(name, func):
    print(f"\n=== 测试 {name} ===")
    start = time.time()
    try:
        result = func()
        print(f"✅ 成功，耗时 {time.time()-start:.2f}s")
        print(f"   结果: {str(result)[:100]}")
    except Exception as e:
        print(f"❌ 失败，耗时 {time.time()-start:.2f}s")
        print(f"   错误: {e}")


# 1. 测试 Ollama
def check_ollama():
    r = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
    return r.json()

# 2. 测试 Neo4j
def check_neo4j():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))
    with driver.session() as s:
        return s.run("RETURN 1 AS n").data()
    driver.close()

# 3. 测试 DeepSeek
def check_deepseek():
    from langchain_deepseek import ChatDeepSeek
    from langchain_core.messages import HumanMessage
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    return llm.invoke([HumanMessage(content="说'ok'")]).content

# 4. 测试完整 Agent
def check_agent():
    from rag_agent import run_conversation
    return run_conversation("根据知识库，加里奥是什么制成的？", thread_id="diag_test")


if __name__ == "__main__":
    test("Ollama", check_ollama)
    test("Neo4j", check_neo4j)
    test("DeepSeek", check_deepseek)
    test("完整 Agent", check_agent)