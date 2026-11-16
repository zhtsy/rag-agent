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
from ingest.clean import cjk_ratio  # noqa: E402
from ingest.loaders import Doc, stats as load_stats  # noqa: E402
from ingest.chunker import stats as chunk_stats  # noqa: E402
from ingest.dedupe import (  # noqa: E402
    dedupe_by_article,
    mark_shadowed_laws,
    drop_toc_blocks,
    supplement_stats,
)

# 图片块的最低汉字占比。0.8 是正常法条的水平，噪声碎片接近 0，中间空得很开；
# 定 0.3 是给"表格页"留余量（表格里数字、符号多，汉字占比本来就不高）。
_IMG_MIN_CJK = 0.3

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _drop_noisy_image_chunks(chunks):
    """丢掉 OCR 认成拉丁字母的图片碎片。返回 (保留的块, 丢弃数)。

    表格块**不参与这条门槛**：一张税率表里汉字本来就少（大半是数字、`%`、`元`），
    汉字占比天然在 0.2 上下，按正文标准会把刚还原出来的表当噪声扔掉 —— 恰好扔掉的
    是图片语料里最有价值的那部分。判据用 markdown 表结构（含 `|`），不猜内容。
    """
    kept, dropped = [], 0
    for c in chunks:
        ext = str((c.metadata or {}).get("ext") or "").lower()
        if ext in _IMAGE_EXTS and "|" not in (c.text or "") and cjk_ratio(c.text) < _IMG_MIN_CJK:
            dropped += 1
            continue
        kept.append(c)
    return kept, dropped


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


def collect(sources, size, overlap, caption=None, with_project_docs=False, dedupe=True):
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
    if s.get("garbage"):
        print(f"⚠️  有 {s['garbage']} 页「有文字层但只是字形编号」（缺 ToUnicode 映射），"
              f"已按空页处理：就算 OCR 救不回来，也绝不能把乱码灌进索引。")
    if s["empty"]:
        print("⚠️  有页解析出来是空的（扫描件且没有 OCR）。它们不会进索引，"
              "但已计入统计 —— 不允许静默消失。")

    chunks = chunk_docs(docs, size=size, overlap=overlap)
    cs = chunk_stats(chunks)
    print(f"✂️  切块: {cs['chunks']} 块, {cs['chars']} 字, "
          f"表格块 {cs['tables']}, 带标题层级 {cs['with_heading']}")

    # 目录块先丢：它没有条号、却塞满「第X章」，问章节内容时会被当成证据召回。
    chunks, toc_dropped = drop_toc_blocks(chunks)
    if toc_dropped:
        print(f"🧹 丢弃目录块: {toc_dropped}")

    # 图片块单独过一道噪声门槛。页级门槛（_load_image 里的 cjk_ratio）只能挡住
    # "整页都是拉丁字母"的情况；一页里大部分是中文、个别横排小字被认错的碎片
    # 会跟着切块漏下来，比如 `AM am, (exo maa, mE.`（20 字，汉字 0 个）。
    # 只对图片做：pdf 那条路已经过了 clean_pages + 字形乱码检测。
    chunks, img_noise = _drop_noisy_image_chunks(chunks)
    if img_noise:
        print(f"🧹 丢弃图片噪声块: {img_noise}（汉字占比 < {_IMG_MIN_CJK}）")

    # 再按「法名 + 条号」去重：同一部法同时有 md 和 pdf 时只留一份（md 优先）。
    # 不去重的话 top-5 会被同义块吃满，答案开始漏。
    if dedupe:
        chunks, ds = dedupe_by_article(chunks)
        print(f"🪢 去重: {ds['before']} → {ds['after']} 块 "
              f"（合并 {ds['dup_groups']} 组、去掉 {ds['dropped']} 块，"
              f"保留非偏好格式独有 {ds['kept_non_preferred']} 块）")
        # 按条去重挡不住"条号没对上"的整部重复，所以再来一道**法级标记**：
        # md 有的法，PDF 块保留但打 shadow=1，检索时 md 同分优先（不删，见
        # ingest/dedupe.mark_shadowed_laws 的说明）。
        #
        # 标记之前先把"还剩什么"量出来：md 漏收的条文、附表页会以 shadow 块形式
        # 留在库里，属于净增量。这张表打在日志里，以后语料变了能立刻看出来。
        ss = supplement_stats(chunks)
        print(f"📐 让位代价体检: md 覆盖 {ss['laws_covered']} 部法；"
              f"pdf 独有条文 {ss['extra_article']} 块、表格 {ss['table_like']} 块、"
              f"无条号散文 {ss['prose']} 块，其余 {ss['dup_article']} 块是同条重复")
        chunks, ls = mark_shadowed_laws(chunks)

        if ls["shadowed"]:
            print(f"🏷️  PDF 降权标记: {ls['shadowed']} 块 "
                  f"（{ls['laws_covered']} 部法已有 md 版本，入库但排在 md 之后）")
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
    ap.add_argument("--no-dedupe", action="store_true",
                    help="不做「法名+条号」去重（压测/对比用）")
    args = ap.parse_args()

    sources = args.src or [
        os.path.join(ROOT, "knowledge_base"),
        os.path.join(ROOT, "knowledge_base_v2"),
    ]
    chunks = collect(sources, args.size, args.overlap,
                     with_project_docs=not args.no_project_docs,
                     dedupe=not args.no_dedupe)
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
    # 记下用的是哪个嵌入模型：换模型重建后忘了重建另一套库时，
    # 查询向量和文档向量会默默跨空间，检索只是变差、不会报错。
    from index_meta import write_signature
    print(f"🔖 嵌入签名: {write_signature(args.out)}")
    print(f"🎉 完成: {args.out}")

    cnt = Counter(d.metadata.get("ext", "?") for d in chunks)
    print("📊 各格式块数:", dict(cnt))
    if args.limit:
        print(f"ℹ️  注意：这是限流索引（{args.limit} 块），只用于压测，不要接入服务")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
