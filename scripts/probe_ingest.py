#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解析探针：把某个目录的文件过一遍解析链路，打印每页/每块的实际情况。

用途是在**不建索引**的前提下回答"这个 PDF/图片到底被读成了什么"。
排查"文件在库里但答不出来"时，第一个要看的就是它。

    python scripts/probe_ingest.py knowledge_base_v2
    python scripts/probe_ingest.py some.pdf --ocr
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ingest import chunk_docs, load_dir, load_file  # noqa: E402
from ingest.loaders import _default_ocr, ocr_available, stats as load_stats  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--ocr", action="store_true", help="启用 tesseract OCR")
    ap.add_argument("--preview", type=int, default=120)
    args = ap.parse_args()

    use_ocr = _default_ocr if args.ocr and ocr_available() else None
    if args.ocr and not ocr_available():
        print("⚠️  未安装 pytesseract，--ocr 无效")

    if os.path.isdir(args.target):
        docs = load_dir(args.target, ocr=use_ocr,
                        on_error=lambda p, e: print(f"❌ {p}: {e}"))
    else:
        docs = load_file(args.target, ocr=use_ocr)

    for d in docs:
        m = d.metadata
        flags = []
        if m.get("empty"):
            flags.append("EMPTY")
        if m.get("needs_ocr"):
            flags.append("NEEDS_OCR")
        if m.get("ocr_used"):
            flags.append("OCR")
        if m.get("caption_used"):
            flags.append("CAPTION")
        tag = ",".join(flags) or "ok"
        page = f":p{m['page']}" if m.get("page") else ""
        print(f"[{tag:12s}] {m.get('source_file','?')}{page} "
              f"kind={m.get('kind')} chars={len(d.text)}")
        if d.text.strip():
            print("    " + d.text.replace("\n", " / ")[: args.preview])

    s = load_stats(docs)
    print(f"\n汇总: {s['pages']} 页, {s['chars']} 字, OCR {s['ocr_used']} 页, 空 {s['empty']} 页")
    if args.ocr is False and s["empty"]:
        print("提示：有空页，加 --ocr 再试一次")

    chunks = chunk_docs(docs)
    tables = [c for c in chunks if c.metadata.get("is_table")]
    print(f"切块: {len(chunks)} 块（表格块 {len(tables)}）")
    for c in tables[:2]:
        print("  表格块 →", c.text.replace("\n", " / ")[:100])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
