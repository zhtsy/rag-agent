#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_ragas_eval.py —— RAGAS 四指标科学评估（云端适配版）

和本地原版的区别
────────────────
1. 原版用 Ollama 的 nomic-embed-text 做 embedding，云端没有 Ollama。
   这里改用项目自己的 `embeddings_shared.LocalEmbeddings`（bge-base-zh），
   和检索链路用的是同一个模型，评估口径更一致。
2. 裁判模型仍然是 DeepSeek（走 API，不占本机算力）。
3. 增加依赖缺失的友好提示——RAGAS 不在基础镜像里，需要单独装。

依赖
────
    pip install ragas datasets          # 约 60-100MB，装在容器里即可

用法
────
    python3 generate_ragas_data.py            # 先生成数据集
    python3 run_ragas_eval.py                 # 再评估
    python3 run_ragas_eval.py --dataset ragas_dataset.json --out ragas_eval_report.csv

注意
────
这个脚本要 import embeddings_shared → 会加载 bge（约 400MB）。
本机只有 3G 内存，**不要和 backend 同时跑重活**；跑之前先确认 free -m 有余量。
"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:
    import pandas as pd
    from datasets import Dataset
    from dotenv import load_dotenv
    from openai import OpenAI
except ImportError as e:  # pragma: no cover
    print(f"❌ 缺少基础依赖：{e}", file=sys.stderr)
    print("   pip install datasets pandas python-dotenv openai", file=sys.stderr)
    sys.exit(1)

try:
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
except ImportError:
    print("❌ 没有安装 RAGAS。", file=sys.stderr)
    print("   容器内安装：docker compose exec -T backend pip install ragas datasets", file=sys.stderr)
    print("   或者写进 requirements.txt 后重建镜像（更持久）。", file=sys.stderr)
    sys.exit(2)


def build_embeddings():
    """优先用项目统一的嵌入入口；失败则明确报错，不静默降级。

    用 get_embeddings() 而不是直接 new LocalEmbeddings：本地跑的是
    Ollama nomic-embed-text（RAG_EMBED_PROVIDER=ollama），直连 bge 会
    拿一个和 chroma 不同分布的空间去算指标，结论无效且不报错。
    """
    try:
        from embeddings_shared import get_embeddings

        print("🔤 使用项目统一嵌入入口（按 RAG_EMBED_PROVIDER 选择）")
        return get_embeddings()
    except Exception as e:  # noqa: BLE001
        print(f"❌ 加载本地嵌入模型失败：{e}", file=sys.stderr)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description="RAGAS 评估（云端适配版）")
    ap.add_argument("--dataset", default="ragas_dataset.json")
    ap.add_argument("--out", default="ragas_eval_report.csv")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--limit", type=int, default=0, help="只评估前 N 条（0=全部）")
    args = ap.parse_args()

    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("❌ 没有 DEEPSEEK_API_KEY，请检查 .env", file=sys.stderr)
        return 1

    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = BASE_DIR / ds_path
    if not ds_path.exists():
        print(f"❌ 找不到数据集：{ds_path}，先跑 generate_ragas_data.py", file=sys.stderr)
        return 1

    import json

    rows = json.loads(ds_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        rows = rows[: args.limit]

    # RAGAS 要求 contexts 是 List[str]，空上下文会让部分指标直接 NaN
    clean = []
    for r in rows:
        ctx = [c for c in (r.get("contexts") or []) if isinstance(c, str) and c.strip()]
        clean.append({
            "question": r["question"],
            "answer": r.get("answer", ""),
            "contexts": ctx or ["（无检索上下文）"],
            "ground_truth": r.get("ground_truth", ""),
        })

    print(f"📦 数据集 {len(clean)} 条 ｜ 裁判模型 {args.model}")

    judge = llm_factory(args.model, client=OpenAI(api_key=api_key, base_url=args.base_url))
    embeddings = build_embeddings()

    result = evaluate(
        dataset=Dataset.from_list(clean),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    print("\n" + "=" * 50)
    print("📊 RAGAS 评估结果")
    print("=" * 50)
    print(result)

    out = Path(args.out)
    if not out.is_absolute():
        out = BASE_DIR / out
    result.to_pandas().to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n✅ 明细已保存：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
