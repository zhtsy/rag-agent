#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""规模曲线：语料从 145 块涨到 3k+ 块，检索召回与延迟怎么变。

为什么这么测
───────────
「库太小」这个质疑真正扎人的地方不是"数据少"，而是"数据少所以随便就能检回来，
你的三路检索没有说服力"。要回答它，只能把同一批问题放到不同规模的语料上跑，
看召回什么时候开始掉。所以这个脚本**只测检索**（金标是答案所在文件），
不让生成模型掺进来——否则模型随机性会盖住规模效应。

口径（写死在这里，免得数据被误读）
──────────────────────────────
- **recall@5**：top-5 里是否包含金标文件。金标有两个来源：
  ① 真实 48 题（test_cases2 + test_cases3）的 `expected_source`
  ② 合成语料自带的 40 条 QA（`knowledge_base_v3/_qa_spec.json`）
- **向量检索**用 numpy 余弦精确算，等价于 chroma 配 cosine 的行为。
  不直接开 chroma 是因为要比 4 个规模点，建 4 个库既慢又难保证同源；
  余弦在这里是纯数学，结果与库实现无关。
- **BM25** 与生产同款：jieba 分词 + BM25Okapi。
- **混合** = RRF(k=60) 融合两路各 top-20 → top-10 →（可选）重排 → top-5，
  与生产链路同构。重排很慢（CPU 约 0.3 s/对），默认只对最后一个规模点跑，
  其余点标 `-`，不假造数字。
- **延迟** 含查询侧开销（查询向量化 / 分词），不含模型加载。

用法：
    python scripts/scale_curve.py                    # 全部规模点
    python scripts/scale_curve.py --sizes 500 3196    # 指定规模点
    python scripts/scale_curve.py --rerank            # 额外跑重排（慢）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CHUNK_CACHE = os.path.join(ROOT, "chroma_db_v3", "_chunks.json")
EMB_CACHE = os.path.join(ROOT, "chroma_db_v3", "_emb.npy")


# ---------------------------------------------------------------- 语料与向量

def collect_chunks(size: int = 200, overlap: int = 40) -> List[dict]:
    """取切块结果，顺序固定为「业务语料 → 合成语料 → 项目文档」。

    顺序为什么重要：子集是**前缀**，取前 N 块时先吃谁，决定了这个点代表什么。

    - 第 1 段 `knowledge_base/`（145 块）＝线上生产语料，可用来给测量本身做校准：
      这个点上 48 题召回应当是 1.0，否则是测量口径错了，不是检索变差了。
    - 第 2 段 `knowledge_base_v3/`＝合成语料，规模曲线的主体。
    - 第 3 段 项目文档（8 篇 md，824 块）放最后。它们单篇极长（README 就有 270 块），
      放前面会把 145 块业务语料淹掉——第一次跑就是这么得出 0.11 的错误结论。
    """
    if os.path.exists(CHUNK_CACHE):
        with open(CHUNK_CACHE, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        import build_kb_v2 as B
        from ingest import load_file
        from ingest.loaders import _default_ocr

        chunks = B.collect(
            [os.path.join(ROOT, "knowledge_base"),
             os.path.join(ROOT, "knowledge_base_v3")],
            size, overlap, with_project_docs=False,
        )
        raw = [{"text": c.text,
                "source_file": c.metadata.get("source_file", "?")}
               for c in chunks]

        # 项目文档单独追加到最后，走同一条解析/切块链路
        tail = []
        for name in B.PROJECT_DOCS:
            path = os.path.join(ROOT, name)
            if os.path.exists(path):
                tail.extend(load_file(path, ocr=_default_ocr))
        if tail:
            from ingest import chunk_docs
            for c in chunk_docs(tail, size=size, overlap=overlap):
                raw.append({"text": c.text,
                            "source_file": c.metadata.get("source_file", "?")})

        os.makedirs(os.path.dirname(CHUNK_CACHE), exist_ok=True)
        with open(CHUNK_CACHE, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)

    # 缓存里只存文本，real 标记在读取时按当次文件集算，避免缓存过期
    for r in raw:
        r["real"] = r["source_file"] in REAL_FILES
    return raw


REAL_FILES: set = set()


def load_real_files() -> set:
    """生产语料的文件名集合（`knowledge_base/` 下的 22 个 txt）。

    刻意不含项目文档：这里的用途是定位"纯生产语料"那个校准点（145 块），
    项目文档在语料顺序里排在最后，不参与这个点。
    """
    files = set()
    kb = os.path.join(ROOT, "knowledge_base")
    if os.path.isdir(kb):
        # 必须递归：语料分在 `food/`、`games/` 等子目录里，只 listdir 顶层
        # 会把 145 块的生产语料误算成 28 块（第一次就是这么错的）。
        for root, _, names in os.walk(kb):
            for f in names:
                files.add(f)
    return files


def embed_all(texts: Sequence[str]) -> np.ndarray:
    """向量化并缓存：3196 块走 ollama 大约几分钟，重跑不该重算。"""
    sig = hashlib.md5("\n".join(texts).encode()).hexdigest()
    if os.path.exists(EMB_CACHE):
        meta_path = EMB_CACHE + ".sig"
        if os.path.exists(meta_path) and open(meta_path).read().strip() == sig:
            print(f"♻️  复用向量缓存 {EMB_CACHE}")
            return np.load(EMB_CACHE)

    from embeddings_shared import get_embeddings

    print(f"🔢 向量化 {len(texts)} 块（首次较慢，之后走缓存）…")
    t0 = time.time()
    vecs = get_embeddings().embed_documents(list(texts))
    arr = np.asarray(vecs, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)   # 归一化 → 点积即余弦
    os.makedirs(os.path.dirname(EMB_CACHE), exist_ok=True)
    np.save(EMB_CACHE, arr)
    open(EMB_CACHE + ".sig", "w").write(sig)
    print(f"   完成 {arr.shape}，用时 {time.time() - t0:.0f}s")
    return arr


# ---------------------------------------------------------------- 检索三路

def rrf(rank_lists: Sequence[Sequence[int]], k: int = 60, top: int = 10) -> List[int]:
    """Reciprocal Rank Fusion：名次加权，不看分数尺度（两路分数不可比）。"""
    score: Dict[int, float] = {}
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])[:top]]


class Index:
    """某个规模点上的检索器：向量 + BM25 + RRF，可选重排。"""

    def __init__(self, chunks: List[dict], embs: np.ndarray):
        import jieba
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.embs = embs
        self.texts = [c["text"] for c in chunks]
        t0 = time.time()
        self.tokens = [list(jieba.cut(t)) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens)
        self.build_sec = time.time() - t0
        self._jieba = jieba

    def vector_top(self, qvec: np.ndarray, k: int = 5) -> Tuple[List[int], float]:
        t0 = time.perf_counter()
        sims = self.embs @ qvec
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return idx.tolist(), (time.perf_counter() - t0) * 1000

    def bm25_top(self, query: str, k: int = 5) -> Tuple[List[int], float]:
        t0 = time.perf_counter()
        scores = self.bm25.get_scores(list(self._jieba.cut(query)))
        idx = np.argpartition(-scores, k)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return idx.tolist(), (time.perf_counter() - t0) * 1000

    def hybrid_top(self, qvec: np.ndarray, query: str,
                   pool: int = 20, top: int = 10) -> Tuple[List[int], float]:
        t0 = time.perf_counter()
        v, _ = self.vector_top(qvec, pool)
        b, _ = self.bm25_top(query, pool)
        fused = rrf([v, b], k=60, top=top)
        return fused, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- 评测

def load_queries(chunks: List[dict]) -> List[dict]:
    """真实 48 题 + 合成 40 题，金标统一成**块下标集合**。

    为什么金标要细到块，而不是文件
    ────────────────────────────
    按文件算命中太松：`test5.txt` 有 6 块，问"小李的猫叫什么"，第 1 块讲的是
    "他在杭州做后端"，答案在第 3 块。向量把第 1 块排进 top-5 就被记成"命中"，
    于是重排按块打分会"把命中挤掉"——那不是重排变差，是度量在骗人。

    所以金标 = **真正含答案的那几块**（路径：`must_contain` 全部命中）。
    这样三路检索与重排用的是同一把尺子。

    ⚠️ 也不能用 `expected_source`：那个字段已经过期（T012 写 `test11.txt`，
    实际文件叫 `kaoya.txt`）。48 题打分从来只看关键词，没人读过这个字段，
    改语料时它烂了也没人发现。
    """
    qs = []
    for name in ("test_cases2.json", "test_cases3.json"):
        for c in json.load(open(os.path.join(ROOT, name), encoding="utf-8")):
            gold = derive_gold(c, chunks)
            if not gold:
                continue        # 关键词跨段导致定不了标：宁可用少而准的，不凑数
            qs.append({"q": c["question"], "gold": gold,
                       "tag": "real", "id": c["id"]})
    spec = os.path.join(ROOT, "knowledge_base_v3", "_qa_spec.json")
    for i, item in enumerate(json.load(open(spec, encoding="utf-8"))["qa"], 1):
        gold = {k for k, c in enumerate(chunks)
                if c["source_file"] == item["gold"] and item["must"] in c["text"]}
        if not gold:
            continue
        qs.append({"q": item["q"], "gold": gold,
                   "tag": "syn", "id": f"S{i:03d}"})
    return qs


def derive_gold(case: dict, chunks: List[dict]) -> set:
    """含该题全部 must_contain 关键词的块下标。

    只在**生产语料**（`knowledge_base/`）里找：项目文档也会出现"喜欢""游戏"
    这类词，把它算成金标会让"小智喜欢什么"指向 README，测出来的召回是假的。
    """
    must = [k.lower() for k in (case.get("must_contain") or []) if k.strip()]
    if not must:
        return set()
    return {k for k, c in enumerate(chunks)
            if c["source_file"] in REAL_FILES
            and all(kw in c["text"].lower() for kw in must)}


def run_point(idx: Index, qvecs: np.ndarray, queries: List[dict],
              pos_of: Dict[int, int], do_rerank: bool = False) -> dict:
    """在一个规模点上跑三路 + 可选重排，返回召回与延迟。

    `pos_of` 把块对象映射回它在**全量语料**里的下标；金标存的是全量下标，
    子集里要还原成局部下标才能比。用 id() 建映射，避免按文本重查。
    """
    hit = {"vector": 0, "bm25": 0, "hybrid": 0, "hybrid_rerank": 0}
    lat = {"vector": [], "bm25": [], "hybrid": [], "rerank": []}
    ctx_chars = []
    per_tag = {t: {"n": 0, **{k: 0 for k in hit}} for t in ("real", "syn")}

    # 子集里"哪些局部下标是金标"
    gold_local = [{k for k, c in enumerate(idx.chunks) if pos_of[id(c)] in q["gold"]}
                  for q in queries]

    reranker = None
    if do_rerank:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(
            (os.getenv("RAG_RERANKER_PATH") or "BAAI/bge-reranker-base").strip(),
            use_fp16=False)

    for qi, item in enumerate(queries):
        gold = gold_local[qi]
        qv = qvecs[qi]

        v, t_v = idx.vector_top(qv, 5)
        b, t_b = idx.bm25_top(item["q"], 5)
        h, t_h = idx.hybrid_top(qv, item["q"], pool=20, top=10)

        ok_v = bool(gold & set(v))
        ok_b = bool(gold & set(b))
        ok_h = bool(gold & set(h))

        lat["vector"].append(t_v)
        lat["bm25"].append(t_b)
        lat["hybrid"].append(t_h)
        ctx_chars.append(sum(len(idx.chunks[i]["text"]) for i in h[:5]))

        ok_r = False
        if reranker is not None:
            t0 = time.perf_counter()
            pairs = [[item["q"], idx.chunks[i]["text"][:512]] for i in h]
            scores = reranker.compute_score(pairs, normalize=True)
            order = np.argsort(-np.asarray(scores))[:5]
            top_r = {h[j] for j in order}
            lat["rerank"].append((time.perf_counter() - t0) * 1000)
            ok_r = bool(gold & top_r)

        for k, ok in (("vector", ok_v), ("bm25", ok_b),
                      ("hybrid", ok_h), ("hybrid_rerank", ok_r)):
            hit[k] += int(ok)
            per_tag[item["tag"]][k] += int(ok)
        per_tag[item["tag"]]["n"] += 1

    n = len(queries)
    out = {
        "chunks": len(idx.chunks),
        "chars": sum(len(t) for t in idx.texts),
        "n_queries": n,
        "recall": {k: round(v / n, 4) for k, v in hit.items()},
        "by_tag": {t: {"n": d["n"],
                       **{k: round(d[k] / max(d["n"], 1), 4)
                          for k in hit}} for t, d in per_tag.items()},
        "latency_ms": {k: (round(float(np.mean(v)), 2) if v else None)
                       for k, v in lat.items()},
        "ctx_chars_top5": int(np.mean(ctx_chars)),
        "build_sec": round(idx.build_sec, 1),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="规模 vs 召回 vs 延迟")
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--rerank", action="store_true", help="跑重排（慢，约 0.3s/对）")
    ap.add_argument("--out", default=os.path.join(ROOT, "scale_curve.json"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    global REAL_FILES
    REAL_FILES = load_real_files()

    chunks = collect_chunks()
    real_n = sum(1 for c in chunks if c["real"])
    print(f"📚 切块总数 {len(chunks)}（真实 {real_n}，合成 {len(chunks) - real_n}）")

    queries = load_queries(chunks)
    n_real = sum(1 for q in queries if q["tag"] == "real")
    print(f"❓ 问题 {len(queries)} 条（真实 {n_real} 条可定标，"
          f"合成 {len(queries) - n_real} 条）")

    vecs = embed_all([c["text"] for c in chunks])
    qvecs = np.asarray(
        __import__("embeddings_shared", fromlist=["get_embeddings"])
        .get_embeddings().embed_documents([q["q"] for q in queries]),
        dtype=np.float32,
    )
    qvecs /= (np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9)

    sizes = args.sizes or [300, 500, 1000, 2000, len(chunks)]

    # ---- 实验设计：固定答案，只增加干扰项 -------------------------------
    # 直接切前缀是错的：金标块在小区间里根本不存在，召回必然是 0，
    # 于是"语料变大 → 召回变好"——那是金标逐渐进库造成的假象，不是检索变强。
    #
    # 正确做法是让**答案永远在库里**，只增加它的竞争者：
    #   锚点 = 所有含答案的块（每块都保留）
    #   干扰 = 其余块，按顺序取 N - len(锚点) 个
    # 这样每个规模点上"5 个槽位要跟多少干扰项抢"才是唯一变量。
    gold_all = set().union(*(q["gold"] for q in queries))
    anchor = [chunks[i] for i in sorted(gold_all)]
    others = [c for i, c in enumerate(chunks) if i not in gold_all]
    print(f"⚓ 锚点 {len(anchor)} 块（含答案），可作干扰 {len(others)} 块")

    pos_of = {id(c): i for i, c in enumerate(chunks)}
    vec_of = {id(c): vecs[i] for i, c in enumerate(chunks)}

    sizes = sorted({max(min(s, len(chunks)), len(anchor)) for s in sizes})
    rows = []
    for n in sizes:
        take = min(n - len(anchor), len(others))
        sub = anchor + others[:take]
        print(f"\n▶ 规模 {len(sub)} 块（干扰项 {take}）")
        idx = Index(sub, np.stack([vec_of[id(c)] for c in sub]))
        row = run_point(idx, qvecs, queries, pos_of, do_rerank=args.rerank)
        row["size"] = len(sub)
        rows.append(row)
        r = row["recall"]
        print(f"   recall@5  向量 {r['vector']:.3f} | BM25 {r['bm25']:.3f} | "
              f"混合 {r['hybrid']:.3f}"
              + (f" | +重排 {r['hybrid_rerank']:.3f}" if args.rerank else ""))
        print(f"   延迟ms    向量 {row['latency_ms']['vector']} | "
              f"BM25 {row['latency_ms']['bm25']} | 混合 {row['latency_ms']['hybrid']}"
              + (f" | 重排 {row['latency_ms']['rerank']}" if args.rerank else ""))
        print(f"   上下文    {row['ctx_chars_top5']} 字/题　"
              f"（真实 {row['by_tag']['real']['hybrid']:.3f} / "
              f"合成 {row['by_tag']['syn']['hybrid']:.3f}）")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "real_chunks": real_n,
                   "n_chunks": len(chunks), "rerank": args.rerank},
                  f, ensure_ascii=False, indent=1)
    print(f"\n💾 {args.out}")

    # 直接吐一张 markdown 表，方便贴进 README
    print("\n| 语料块数 | 字数量 | 向量 | BM25 | 混合 | 混合+重排 | 混合延迟 | 上下文 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        rec = r["recall"]
        rr = f"{rec['hybrid_rerank']:.3f}" if args.rerank else "-"
        print(f"| {r['chunks']} | {r['chars']:,} | {rec['vector']:.3f} | "
              f"{rec['bm25']:.3f} | {rec['hybrid']:.3f} | {rr} | "
              f"{r['latency_ms']['hybrid']} ms | {r['ctx_chars_top5']} 字 |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
