#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""建成「第二套索引」：多格式语料 → 独立目录，与生产索引物理隔离。

为什么要另开一套而不是改 build_knowledge_base.py
─────────────────────────────────────────────
生产索引（`chroma_db/`）是 48 题评测的基线，重建一次就要重跑一次分。
这次实验的结果可能不如预期，所以**旧索引一个字节都不动**：
新语料进 `knowledge_base_v2/`，新索引进 `chroma_db_v2/`，
要切回去只要不设置 `RAG_CHROMA_DIR` 就行。

用法：
    python scripts/build_kb_v2.py --dry-run              # 只统计，不建库
    python scripts/build_kb_v2.py                        # 建 chroma_db_v2
    python scripts/build_kb_v2.py --out /tmp/chroma_x    # 建到别处（压测用）
    python scripts/build_kb_v2.py --limit 500            # 只取前 500 块（规模曲线用）
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ingest import chunk_docs, load_dir  # noqa: E402
from ingest.loaders import Doc, stats as load_stats  # noqa: E402
from ingest.chunker import stats as chunk_stats  # noqa: E402


def build_ocr():
    """有 tesseract 就返回 OCR 函数，没有就返回 None（并明确告知）。"""
    try:
        import pytesseract  # noqa: F401
        from ingest.loaders import _default_ocr
        return _default_ocr
    except Exception:
        return None


# 项目自身的文档也当语料：真实、有噪声、有版本演进，而且"问自己项目"是
# 最容易验证对错的场景（我知道正确答案）。直接读仓库根目录，不复制副本，
# 避免同一份 184 KB 文本在仓库里存两遍。
PROJECT_DOCS = [
    "README.md", "CASE_STUDY.md", "OPS.md", "MIGRATION.md",
    "RESULTS.md", "PERF.md", "VARIANCE.md", "MCP.md",
]


def collect(sources, size, overlap, caption=None, with_project_docs=False):
    from ingest import load_file

    ocr = build_ocr()
    docs = []
    if with_project_docs:
        for name in PROJECT_DOCS:
            path = os.path.join(ROOT, name)
            if not os.path.exists(path):
                continue
            got = load_file(path, ocr=ocr, caption=caption)
            docs.extend(got)
        print(f"📂 项目文档: {len(PROJECT_DOCS)} 个 md")

    for src in sources:
        if not os.path.isdir(src):
            print(f"⚠️  跳过不存在的目录: {src}")
            continue
        got = load_dir(src, ocr=ocr, caption=caption,
                       on_error=lambda p, e: print(f"❌ {p}: {type(e).__name__}: {e}"))
        print(f"📂 {src}: {len(got)} 页/段")
        docs.extend(got)

    s = load_stats(docs)
    print(f"\n📥 加载汇总: {s['pages']} 页, {s['chars']} 字, "
          f"OCR 救回 {s['ocr_used']} 页, 空页 {s['empty']} 页")
    if s["empty"]:
        print("⚠️  有页解析出来是空的（扫描件且没有 OCR）。它们不会进索引，"
              "但已计入统计 —— 不允许静默消失。")

    chunks = chunk_docs(docs, size=size, overlap=overlap)
    cs = chunk_stats(chunks)
    print(f"✂️  切块: {cs['chunks']} 块, {cs['chars']} 字, "
          f"表格块 {cs['tables']}, 带标题层级 {cs['with_heading']}")
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="多格式第二套索引构建")
    ap.add_argument("--src", action="append", default=None,
                    help="语料目录，可多次给。默认 knowledge_base + knowledge_base_v2")
    ap.add_argument("--out", default=os.path.join(ROOT, "chroma_db_v2"))
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 块（规模曲线用）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-project-docs", action="store_true",
                    help="不把仓库根目录的项目文档算进语料（只跑样本）")
    args = ap.parse_args()

    sources = args.src or [
        os.path.join(ROOT, "knowledge_base"),
        os.path.join(ROOT, "knowledge_base_v2"),
    ]
    chunks = collect(sources, args.size, args.overlap,
                     with_project_docs=not args.no_project_docs)
    if args.limit:
        chunks = chunks[: args.limit]
        print(f"✂️  限流后: {len(chunks)} 块")

    if args.dry_run:
        print("🧪 dry-run，未写入任何索引")
        return 0

    from langchain_chroma import Chroma

    from embeddings_shared import get_embeddings

    if os.path.isdir(args.out) and os.listdir(args.out):
        print(f"♻️  目标已存在，先清空: {args.out}")
        import shutil
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    from langchain_core.documents import Document

    lc_docs = [
        Document(page_content=c.text, metadata={k: v for k, v in c.metadata.items()
                                                if isinstance(v, (str, int, float, bool))})
        for c in chunks
    ]
    print(f"🔢 向量化 {len(lc_docs)} 块…")
    Chroma.from_documents(documents=lc_docs, embedding=get_embeddings(),
                          persist_directory=args.out)
    print(f"🎉 完成: {args.out}")

    cnt = Counter(d.metadata.get("ext", "?") for d in chunks)
    print("📊 各格式块数:", dict(cnt))
    if args.limit:
        print(f"ℹ️  注意：这是限流索引（{args.limit} 块），只用于压测，不要接入服务")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
