#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""print_all_chunks.py —— 把向量库里的块全部打印出来，人工核对分块质量。

和本地原版的区别
────────────────
原版 `from rag_agent import vectorstore` 会连带加载 bge + reranker（约 1.5G），
只是为了看几行文本而已，代价太大。这里直接用 chromadb 读持久化目录，
**不加载任何模型**。

用法
────
    python3 print_all_chunks.py                 # 全部打印，内容截断到 300 字
    python3 print_all_chunks.py --limit 10      # 只看前 10 个
    python3 print_all_chunks.py --grep 小陈      # 只看内容含某个词的块
    python3 print_all_chunks.py --full          # 不截断
"""

import argparse
from pathlib import Path

import chromadb

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION = "langchain"


def main() -> None:
    ap = argparse.ArgumentParser(description="打印向量库分块")
    ap.add_argument("--dir", default=str(PERSIST_DIR))
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--grep", default="")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--show-embedding", action="store_true", help="打印向量维度（不打印向量本身）")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.dir)
    names = [c.name for c in client.list_collections()]
    if args.collection not in names:
        print(f"❌ 找不到集合 {args.collection}，现有：{names}")
        return

    col = client.get_collection(args.collection)
    total = col.count()
    print(f"📦 集合 {args.collection}｜共 {total} 个块｜目录 {args.dir}\n")

    got = col.get(include=["documents", "metadatas"])
    ids, docs = got["ids"], got["documents"]
    metas = got["metadatas"] or [{} for _ in ids]

    shown = 0
    for i, (cid, doc, meta) in enumerate(zip(ids, docs, metas), 1):
        if args.grep and args.grep not in (doc or ""):
            continue
        shown += 1
        if args.limit and shown > args.limit:
            break

        meta = meta or {}
        body = doc or ""
        text = body if args.full else body[:300]
        print(f"========== 块 {shown} / {total} ==========")
        print(f"🆔 ID: {cid}")
        print(f"📁 来源: {meta.get('source', '未知')}")
        print(f"👤 实体: {meta.get('entity', meta.get('person', '无'))}")
        print(f"📏 长度: {len(body)} 字")
        print(f"📝 内容: {text}{'' if args.full or len(body) <= 300 else '...'}")
        print()

    if args.grep and shown == 0:
        print(f"🔍 没有块的正文包含「{args.grep}」")

    if args.show_embedding and ids:
        one = col.get(ids=[ids[0]], include=["embeddings"])
        emb = one["embeddings"]
        if emb is not None and len(emb):
            print(f"🔢 向量维度：{len(emb[0])}")


if __name__ == "__main__":
    main()
