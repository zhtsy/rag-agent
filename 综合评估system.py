import json
import time
import os
from rag_agent import run_conversation
from langchain_ollama import OllamaEmbeddings
import numpy as np

# ========== 1. 初始化嵌入模型 ==========
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)


# ========== 2. 语义相似度计算 ==========
def semantic_similarity(text1: str, text2: str) -> float:
    """计算两段文本的语义相似度（0-1）"""
    emb1 = embeddings.embed_query(text1)
    emb2 = embeddings.embed_query(text2)
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(sim)


# ========== 3. 来源准确率计算（兼容列表和字符串） ==========
def source_accuracy(sources: list, expected_source) -> float:
    """
    按召回率计算来源准确率：
    命中的期望来源数 / 期望来源总数
    - expected_source 可以是字符串、列表或空字符串
    - 如果期望来源为空，返回 1.0（拒答类不扣分）
    """
    if not expected_source:
        return 1.0  # 拒答类，不要求来源

    # 统一转成列表
    if isinstance(expected_source, list):
        expected_sources = expected_source
    else:
        expected_sources = [expected_source]

    if not sources:
        return 0.0

    # 统计命中的期望来源数
    hit_count = 0
    for es in expected_sources:
        for s in sources:
            if es in s:
                hit_count += 1
                break  # 避免同一个期望来源被重复计数

    return hit_count / len(expected_sources)



# ========== 4. 主评估函数 ==========
def evaluate():
    # 加载测试用例
    with open("test_cases.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []

    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"测试 {case['id']}: {case['question']}")

        # 计时
        start = time.time()
        result = run_conversation(case["question"], thread_id=case["id"])
        elapsed = time.time() - start

        # 提取结果
        if hasattr(result, "answer"):
            answer = result.answer
            sources = result.sources
            confidence = result.confidence
        else:
            answer = str(result)
            sources = []
            confidence = 0.0

        # ---- 维度一：语义相似度 ----
        sim = semantic_similarity(answer, case["expected_answer"])
        semantic_score = 1.0 if sim > 0.8 else sim

        # ---- 维度二：来源准确率 ----
        source_score = source_accuracy(sources, case.get("expected_source", ""))

        # ---- 维度三：禁止词检查 ----
        must_not = case.get("must_not_contain", [])
        violation = sum(1 for kw in must_not if kw in answer)
        violation_rate = violation / len(must_not) if must_not else 0.0

        # ---- 综合得分 ----
        score = semantic_score * (1 - violation_rate)

        print(f"  答案: {answer[:80]}...")
        print(f"  语义相似度: {sim:.3f}")
        print(f"  来源准确: {'✅' if source_score else '❌'}")
        print(f"  置信度: {confidence}")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  得分: {score:.3f}")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "score": score,
            "semantic_similarity": sim,
            "source_accuracy": source_score,
            "confidence": confidence,
            "elapsed": elapsed,
            "correct": score > 0.7
        })

    # ========== 汇总统计 ==========
    avg_score = sum(r["score"] for r in results) / len(results)
    avg_sim = sum(r["semantic_similarity"] for r in results) / len(results)
    avg_source = sum(r["source_accuracy"] for r in results) / len(results)
    avg_time = sum(r["elapsed"] for r in results) / len(results)

    print(f"\n{'='*50}")
    print(f"📊 评估报告")
    print(f"{'='*50}")
    print(f"总测试数: {len(results)}")
    print(f"平均综合得分: {avg_score:.3f}")
    print(f"平均语义相似度: {avg_sim:.3f}")
    print(f"平均来源准确率: {avg_source:.3f}")
    print(f"平均耗时: {avg_time:.2f}s")

    # 按类别统计
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r["score"])
    print(f"\n📊 分类统计:")
    for cat, scores in categories.items():
        print(f"  {cat}: 平均 {sum(scores)/len(scores):.3f} ({len(scores)}个)")

    # ========== 置信度校准 ==========
    buckets = {"0.0-0.5": [], "0.5-0.7": [], "0.7-0.9": [], "0.9-1.0": []}
    for r in results:
        conf = r["confidence"]
        if conf < 0.5:
            buckets["0.0-0.5"].append(r["correct"])
        elif conf < 0.7:
            buckets["0.5-0.7"].append(r["correct"])
        elif conf < 0.9:
            buckets["0.7-0.9"].append(r["correct"])
        else:
            buckets["0.9-1.0"].append(r["correct"])

    print(f"\n📊 置信度校准:")
    for bucket, correctness in buckets.items():
        if correctness:
            acc = sum(correctness) / len(correctness)
            print(f"  {bucket}: 实际正确率 {acc:.2%} ({len(correctness)}个)")
        else:
            print(f"  {bucket}: 无样本")

    # ========== 保存报告 ==========
    report = {
        "avg_score": avg_score,
        "avg_semantic_similarity": avg_sim,
        "avg_source_accuracy": avg_source,
        "avg_elapsed": avg_time,
        "total": len(results),
        "details": results
    }
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 报告已保存到 eval_report.json")


if __name__ == "__main__":
    evaluate()