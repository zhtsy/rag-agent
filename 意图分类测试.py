from rag_agent import run_conversation

# 关系查询 → 应该走图谱
print(run_conversation("根据知识库，小陈住在哪个省？", thread_id="t1"))

# 变体关系查询 → 也应该走图谱（关键词没命中，但语义命中）
print(run_conversation("根据知识库，加里奥是谁造的？", thread_id="t2"))

# 因果解释 → 应该走向量
print(run_conversation("根据知识库，加里奥为什么能吸收魔法？", thread_id="t3"))

# 普通属性查询 → 应该走向量
print(run_conversation("根据知识库，小张擅长什么？", thread_id="t4"))