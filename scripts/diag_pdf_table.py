# -*- coding: utf-8 -*-
"""看一份 PDF 从解析到切块，表格到底经历了什么。

为什么要这个
────────────
`_load_pdf` 对一页做四件事，任何一步都能把表格毁掉，而且**毁掉之后表面上
一切正常**（页有文字、块数不少、甚至还能搜到几个词）：

  1. pdfplumber 抽表（lines 策略，靠框线）—— 抽不到就没有表；
  2. 双栏重排（`_pdf_layout_text`）—— 表格中间那列空白会被当成"栏间带"，
     把表拦腰切成左右两半（现在含表的页已排除在重排之外）；
  3. 单元格取字（`_chars_to_text` / `_vertical_cell_text`）—— 竖排、合并单元格、
     数字与汉字基线不齐，三种情况都能让"每个字都对、顺序全错"；
  4. 切块（`chunker._split_table`）—— 按正文字数切表会把 17 行切成 14 块。

所以逐页打印：抽到几张表、前几行长什么样、这页有没有被 relayout、
解析出的正文里表还在不在，以及切块后 `is_table` 有几块。

用法
────
    python3 scripts/diag_pdf_table.py knowledge_base_pdf/中华人民共和国车船税法_*.pdf
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import loaders                                    # noqa: E402
from ingest.chunker import chunk_doc                          # noqa: E402


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]

    print(f"== {os.path.basename(path)}")
    tables = loaders._pdf_tables_md(path)
    print(f"pdfplumber 抽到表格的页：{sorted(tables)}")
    for pno, mds in sorted(tables.items()):
        for k, md in enumerate(mds, 1):
            rows = [r for r in md.splitlines() if r.strip()]
            print(f"  p{pno} 表{k}: {len(rows)} 行")
            for r in rows[:4]:
                print(f"      {r[:110]}")

    docs = loaders.load_file(path, ocr=loaders._default_ocr)
    print(f"\n解析出 {len(docs)} 页")
    for d in docs:
        m = d.metadata
        flags = []
        for key in ("relayout", "ocr_used", "garbage", "empty", "needs_ocr", "tables"):
            if m.get(key):
                flags.append(f"{key}={m[key]}")
        head = d.text.replace("\n", " ⏎ ")[:90]
        print(f"  p{m.get('page')}: {len(d.text)} 字  {' '.join(flags)}")
        print(f"      {head}")

    chunks = []
    for d in docs:
        chunks.extend(chunk_doc(d))
    n_tbl = sum(1 for c in chunks if c.metadata.get("is_table"))
    print(f"\n切出 {len(chunks)} 块，其中 is_table={n_tbl}")
    for c in chunks:
        if c.metadata.get("is_table"):
            first = c.text.splitlines()[0][:90]
            print(f"  [table] {len(c.text)} 字  {first}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
