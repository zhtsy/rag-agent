import json
import time
import gc
import torch
from rag_agent import run_conversation


def evaluate():
    with open("test_cases2.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    total_score = 0

    for case in test_cases:
        print(f"\n{'=' * 50}")
        print(f"测试 {case['id']}: {case['question']}")

        start = time.time()

        try:
            result = run_conversation(case["question"], thread_id=case["id"])
        except Exception as e:
            print(f"❌ 测试 {case['id']} 发生异常: {e}")
            result = None

        elapsed = time.time() - start

        # 【修改点】兼容处理：支持 AgentAnswer 对象或纯字符串，防止误杀拒答类
        if result and hasattr(result, "answer"):
            answer = result.answer
            sources = result.sources
            confidence = result.confidence
        elif isinstance(result, str):
            answer = result
            sources = []
            confidence = 0.0
        else:
            answer = "执行异常或返回空"
            sources = []
            confidence = 0.0

        # 评分：关键词命中
        if case["category"] == "拒答":
            hit = 1 if any(kw in answer for kw in case["must_contain"]) else 0
            hit_rate = float(hit)
        else:
            hit = sum(1 for kw in case["must_contain"] if kw in answer)
            hit_rate = hit / len(case["must_contain"]) if case["must_contain"] else 1.0

        # 评分：禁止词检查
        violation = sum(1 for kw in case["must_not_contain"] if kw in answer)
        violation_rate = violation / len(case["must_not_contain"]) if case["must_not_contain"] else 0.0

        score = hit_rate * (1 - violation_rate)
        total_score += score

        print(f"  答案: {answer[:80]}...")
        print(f"  来源: {sources}")
        print(f"  置信度: {confidence}")
        print(f"  关键词命中: {hit}/{len(case['must_contain'])}")
        print(f"  禁止词命中: {violation}/{len(case['must_not_contain'])}")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  得分: {score:.2f}")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "score": score,
            "hit_rate": hit_rate,
            "violation_rate": violation_rate,
            "confidence": confidence,
            "elapsed": elapsed,
            "answer": answer[:100]
        })

        # 强制进行内存和显存清理
        print("  🧹 正在清理内存和显存...")
        if 'result' in locals():
            del result

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        time.sleep(3)

    # 生成报告
    avg_score = total_score / len(test_cases)
    print(f"\n{'=' * 50}")
    print(f"📊 评估报告")
    print(f"{'=' * 50}")
    print(f"总测试数: {len(test_cases)}")
    print(f"平均得分: {avg_score:.2f}")

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["score"])
    for cat, scores in categories.items():
        print(f"  {cat}: 平均 {sum(scores) / len(scores):.2f} ({len(scores)}个)")

    with open("eval_report6.json", "w", encoding="utf-8") as f:
        json.dump({
            "avg_score": avg_score,
            "total": len(test_cases),
            "details": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 报告已保存到 eval_report6.json")


if __name__ == "__main__":
    evaluate()