#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照实验：把整个知识库塞进上下文 vs 用检索取 top-k。

要回答的那句话
─────────────
"你库才两万字、145 个块，全塞进去就行了，为什么要搭三路检索加重排？"

这个实验把问题变成数据：**同一个模型、同一套提示词，唯一变量是上下文从哪来**。
  A 臂：全库 19,789 字原样塞进去（约 1.3 万 token）
  B 臂：走真实检索链路（意图 → 图谱 → 混合检索 → 重排 → top-5）

三组指标：得分（eval_system.score_case）、单题耗时、单题上下文 token。

诚实的预期：**小库上 A 臂很可能不输**，甚至更高——因为不存在"召回失败"。
那结论就不是"RAG 更好"，而是"RAG 的必要性来自规模与成本，不是小库的正确率"。
这个结论比硬吹 RAG 有用得多，也是面试时能站住的那一版。

用法：
    python scripts/ab_fullcontext.py --limit 8      # 先跑 8 题看看
    python scripts/ab_fullcontext.py                # 48 题全跑
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SYSTEM = """你是知识库问答助手。只能依据下面提供的资料回答，不许编造。
资料里没有的，直接回答"无法从知识库中找到相关信息。"。回答要简短、直接。"""


def corpus_text(persist_dir: str) -> Tuple[str, int]:
    """把整个库的块按顺序拼成一段文本。"""
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    names = [c.name for c in client.list_collections()]
    if not names:
        raise SystemExit(f"{persist_dir} 里没有集合")
    col = client.get_collection(names[0])
    data = col.get(include=["documents", "metadatas"])
    docs = data["documents"] or []
    metas = data["metadatas"] or [{} for _ in docs]
    # 按来源文件分组，读起来更像"一本资料"
    grouped: dict[str, List[str]] = {}
    for d, m in zip(docs, metas):
        src = (m or {}).get("source_file") or "unknown"
        grouped.setdefault(src, []).append(d)
    parts = []
    for src, chunks in grouped.items():
        parts.append(f"【{src}】\n" + "\n".join(chunks))
    return "\n\n".join(parts), len(docs)


def retrieve_context(query: str) -> str:
    """走真实检索链路。导入失败时退化成纯向量 top-5，并在结果里标明。"""
    try:
        import rag_agent
        return rag_agent._search_knowledge_base_impl(query=query, original_query=query)
    except Exception as exc:  # pragma: no cover - 依赖环境
        print(f"⚠️  真实检索不可用（{type(exc).__name__}: {exc}），退化为向量 top-5")
        import chromadb

        from embeddings_shared import get_embeddings

        client = chromadb.PersistentClient(path=os.path.join(ROOT, "chroma_db"))
        col = client.get_collection(client.list_collections()[0].name)
        emb = get_embeddings().embed_query(query)
        res = col.query(query_embeddings=[emb], n_results=5)
        return "\n\n".join(res["documents"][0])


def make_llm():
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(model="deepseek-chat", temperature=0, timeout=120, max_retries=3)


def ask(llm, context: str, question: str) -> Tuple[str, float]:
    t0 = time.time()
    msg = [
        ("system", SYSTEM),
        ("human", f"资料：\n{context}\n\n问题：{question}\n\n请只依据资料回答。"),
    ]
    out = llm.invoke(msg)
    return (out.content or "").strip(), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+",
                    default=[os.path.join(ROOT, "test_cases2.json"),
                             os.path.join(ROOT, "test_cases3.json")])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="只跑 A 或 B")
    ap.add_argument("--out", default=os.path.join(ROOT, "ab_fullcontext.json"))
    ap.add_argument("--full-dir", default=os.path.join(ROOT, "chroma_db"),
                    help="A 臂用哪个索引的全文（默认生产库；压测时指向 chroma_db_v2）")
    args = ap.parse_args()

    from dotenv import load_dotenv

    # 必须显式加载：rag_agent 自己在模块头 load_dotenv，但这里的 llm 在
    # import rag_agent 之前就构造了，读不到 DEEPSEEK_API_KEY 会直接报错。
    load_dotenv(os.path.join(ROOT, ".env"))

    from eval_system import load_cases, score_case

    cases, _loaded = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    full, n_chunks = corpus_text(args.full_dir)
    print(f"📚 全库上下文: {n_chunks} 块 / {len(full)} 字 "
          f"（约 {len(full) / 1.6:.0f} token，按 1.6 字/token 粗估）")

    llm = make_llm()
    rows = []
    for i, case in enumerate(cases, start=1):
        q = case["question"]
        rec = {"id": case.get("id"), "question": q}

        if args.only != "B":
            ans_a, dt_a = ask(llm, full, q)
            s_a = score_case(case, ans_a)[0]
            rec.update(a_score=s_a, a_sec=round(dt_a, 1),
                       a_ctx_chars=len(full), a_answer=ans_a[:120])

        if args.only != "A":
            ctx = retrieve_context(q)
            ans_b, dt_b = ask(llm, ctx, q)
            s_b = score_case(case, ans_b)[0]
            rec.update(b_score=s_b, b_sec=round(dt_b, 1),
                       b_ctx_chars=len(ctx), b_answer=ans_b[:120])

        rows.append(rec)
        print(f"[{i}/{len(cases)}] {rec['id']:5s} "
              f"A={rec.get('a_score', '-')} ({rec.get('a_sec', '-')}s, "
              f"{rec.get('a_ctx_chars', 0)}字) | "
              f"B={rec.get('b_score', '-')} ({rec.get('b_sec', '-')}s, "
              f"{rec.get('b_ctx_chars', 0)}字)")

    def avg(key):
        vals = [r[key] for r in rows if key in r]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "n": len(rows),
        "a_score": avg("a_score"), "b_score": avg("b_score"),
        "a_sec": avg("a_sec"), "b_sec": avg("b_sec"),
        "a_ctx_chars": avg("a_ctx_chars"), "b_ctx_chars": avg("b_ctx_chars"),
    }
    print("\n=== 汇总 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"明细已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
