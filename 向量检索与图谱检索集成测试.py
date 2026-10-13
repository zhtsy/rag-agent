from rag_agent import run_conversation

# # 测试多跳推理（应该走图谱）
# result = run_conversation("小陈住在哪个省？", thread_id="test_graph_1")
# print(result)
#
# # 测试普通检索（应该走向量库）
# result = run_conversation("小张擅长做什么甜点？", thread_id="test_graph_2")
# print(result)
#
# # 测试图谱没有的关系（应该回退到向量库或拒答）
# result = run_conversation("小智的银行卡密码是多少？", thread_id="test_graph_3")
# print(result)

result = run_conversation("根据知识库，加里奥为什么能吸收魔法？", thread_id="test_graph_4")
print(result)
