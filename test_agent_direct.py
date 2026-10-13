from rag_agent import run_conversation

print("=== 开始测试 ===")
result = run_conversation("根据知识库，加里奥是什么制成的？", thread_id="direct_test")
print("=== 测试结束 ===")
print(result)