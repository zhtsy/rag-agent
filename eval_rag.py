# eval_rag.py
from rag本地检索助手 import run_conversation  # 导入你的主函数

test_cases = [
    {"question": "我的名字是什么？", "expected_phrase": "小智"},
    {"question": "我住在哪里？", "expected_phrase": "南京"},
    {"question": "我喜欢什么编程语言？", "expected_phrase": "Python"},
]

def evaluate():
    correct = 0
    total = len(test_cases)
    for case in test_cases:
        question = case["question"]
        expected = case["expected_phrase"]
        print(f"提问: {question}")
        answer = run_conversation(question)
        print(f"回答: {answer}")
        if expected in answer:
            correct += 1
            print("✅ 通过")
        else:
            print(f"❌ 未通过（期望包含 '{expected}'）")
        print("-" * 40)
    print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")

if __name__ == "__main__":
    evaluate()