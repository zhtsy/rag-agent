#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在第二套索引上验证：多格式内容能否被"问题"召回。

结论会直接写进 README，所以这里不只报一个总数，而是**把三条路拆开对比**：
    vector 纯向量 / bm25 纯关键词 / hybrid 两路 RRF 融合

为什么值得拆开：OCR 出来的文本是碎片（`402000, 431000, 451300, Jul, Aug`），
语义向量几乎抓不住"哪个月最高"；但它对关键词是友好的。反过来，
"毛利率是多少"这种问法和原文措辞不同，关键词命不中，向量能兜住。
单路一定有一边漏，这正是混合检索存在的理由——而且在小语料上也成立。
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CASES = [
    ("2026 年第三季度营业收入是多少？", "q3_jianbao_scan.pdf"),
    ("毛利率是多少，环比变化多少？", "q3_jianbao_scan.pdf"),
    ("川菜和粤菜的口味差别是什么？", "caixi_duizhao.pdf"),
    ("哪个月营收最高，金额多少？", "q3_revenue_chart.png"),
    ("营收的统计口径是什么？", "jingying_zhibiao.md"),
]


def tokenize(text: str):
    import jieba

    words = [w.strip().lower() for w in jieba.lcut(text) if w.strip()]
    # 中文单字信息量太低，纯符号丢掉
    return [w for w in words if len(w) > 1 or re.match(r"[a-z0-9]", w)]


def rrf(rankings, k: int = 60):
    """Reciprocal Rank Fusion：只用名次，不用原始分数（两路分数量纲不同）。"""
    score = {}
    for ranked in rankings:
        for pos, idx in enumerate(ranked):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])]


def main() -> int:
    import chromadb
    from rank_bm25 import BM25Okapi

    from embeddings_shared import get_embeddings

    path = os.path.join(ROOT, "chroma_db_v2")
    if not os.path.isdir(path):
        raise SystemExit("先跑 python scripts/build_kb_v2.py")

    client = chromadb.PersistentClient(path=path)
    col = client.get_collection(client.list_collections()[0].name)
    data = col.get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]
    id2idx = {cid: i for i, cid in enumerate(data["ids"])}
    srcs = [(m or {}).get("source_file", "?") for m in metas]
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    emb = get_embeddings()

    # 生产链路是「混合召回 10 → 重排 → 取 5」，这里把最后一环也加上，
    # 否则测的不是线上那套。重排对"两路都召回但排序不对"的情况最有效。
    from FlagEmbedding import FlagReranker

    reranker = FlagReranker(
        os.getenv("RAG_RERANKER_PATH") or "BAAI/bge-reranker-base", use_fp16=False
    )

    stat = {"vector": 0, "bm25": 0, "hybrid": 0, "rerank": 0}
    for q, expect in CASES:
        # 1) 向量
        vres = col.query(query_embeddings=[emb.embed_query(q)], n_results=10,
                         include=["metadatas"])
        vids = [id2idx[i] for i in vres["ids"][0] if i in id2idx]
        # 2) BM25
        scores = bm25.get_scores(tokenize(q))
        bids = sorted(range(len(docs)), key=lambda i: -scores[i])[:10]
        # 3) 融合
        hids = rrf([vids, bids])[:3]
        # 4) 重排：混合召回的前 10 个候选交给 bge-reranker
        cand = rrf([vids, bids])[:10]
        pairs = [[q, docs[i]] for i in cand]
        scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        rids = [cand[i] for i in sorted(range(len(cand)), key=lambda i: -scores[i])[:3]]

        v_top = [srcs[i] for i in vids[:3]]
        b_top = [srcs[i] for i in bids[:3]]
        h_top = [srcs[i] for i in hids]
        r_top = [srcs[i] for i in rids]
        hit_v, hit_b, hit_h = expect in v_top, expect in b_top, expect in h_top
        hit_r = expect in r_top
        stat["vector"] += hit_v
        stat["bm25"] += hit_b
        stat["hybrid"] += hit_h
        stat["rerank"] += hit_r

        print(f"\n{q}")
        print(f"  vector  {'✅' if hit_v else '❌'} {v_top}")
        print(f"  bm25    {'✅' if hit_b else '❌'} {b_top}")
        print(f"  hybrid  {'✅' if hit_h else '❌'} {h_top}")
        print(f"  +rerank {'✅' if hit_r else '❌'} {r_top}")

    n = len(CASES)
    print(f"\n=== 召回（top-3 命中预期文件）===")
    print(f"  vector-only   : {stat['vector']}/{n}")
    print(f"  bm25-only     : {stat['bm25']}/{n}")
    print(f"  hybrid(RRF)   : {stat['hybrid']}/{n}")
    print(f"  hybrid+rerank : {stat['rerank']}/{n}")
    return 0 if stat["rerank"] == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
