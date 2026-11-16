# -*- coding: utf-8 -*-
"""盘点 PDF 里的表格：哪些页有表、抽出来是什么样、默认策略漏没漏。

为什么要单独盘
──────────────
`loaders._pdf_tables_md` 一直用 pdfplumber 的**默认策略**（靠页面上的横竖线
找单元格）。这个策略对"公文里插一张有线表"够用，但对**扫描件附表**和
**只有横线的税表**会一无所获 —— 而这类表恰恰是税法 PDF 唯一有价值的部分。
不盘一遍就不知道到底是"语料里没表"还是"抽取没抽到"，容易改错方向。

判据
────
对每页跑两次：
  · `lines` 策略（默认）：要靠框线；
  · `text` 策略：靠字块对齐，不需要线。
再算这页的「数字密度」和「短行比例」，作为"这页本来就该是表吗"的先验。
`verdict` 三态：
  · `lines`  / `text` —— 该策略抽到了可信的表；
  · `suspect` —— 看着像表但两种策略都没抽出来（要 OCR 或人工看）；
  · `none` —— 不像表。

用法
────
    python3 scripts/probe_tables.py                      # 全目录
    python3 scripts/probe_tables.py --only 资源税 车船税
    python3 scripts/probe_tables.py --json /tmp/t.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(ROOT, "knowledge_base_pdf")

TEXT_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "min_words_vertical": 2,
    "min_words_horizontal": 1,
}


def page_signals(text: str) -> dict:
    """数字密度 + 短行比例。表格页的这两个值都明显高于条款正文。"""
    body = re.sub(r"\s+", "", text or "")
    digits = sum(ch.isdigit() for ch in body)
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    short = sum(1 for ln in lines if len(ln) <= 12)
    return {
        "chars": len(body),
        "digit_density": round(digits / max(1, len(body)), 3),
        "short_ratio": round(short / max(1, len(lines)), 3),
    }


def table_quality(rows) -> tuple[int, int]:
    rows = [r for r in (rows or []) if any((c or "").strip() for c in r)]
    cols = max((len(r) for r in rows), default=0)
    return len(rows), cols


def probe(path: str) -> dict:
    import pdfplumber

    out = {"file": os.path.basename(path), "pages": []}
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            sig = page_signals(text)
            if sig["chars"] < 20:
                out["pages"].append({"page": i, "verdict": "empty", **sig})
                continue
            best = None
            for name, settings in (("lines", None), ("text", TEXT_SETTINGS)):
                try:
                    tables = page.extract_tables(table_settings=settings) or []
                except Exception:                   # noqa: BLE001
                    tables = []
                cand = None
                for rows in tables:
                    nr, nc = table_quality(rows)
                    # 至少 3 行 2 列才算表；2 行 2 列的多半是"两行正文被当成表"
                    if nr >= 3 and nc >= 2:
                        if cand is None or nr * nc > cand[0] * cand[1]:
                            cand = (nr, nc, rows)
                if cand:
                    best = {"strategy": name, "rows": cand[0], "cols": cand[1],
                            "sample": [" | ".join((c or "").replace("\n", " ")
                                                  for c in r)[:120]
                                       for r in cand[2][:4]]}
                    break
            if best:
                verdict = best["strategy"]
            elif sig["digit_density"] >= 0.12 and sig["short_ratio"] >= 0.5:
                verdict = "suspect"
            else:
                verdict = "none"
            entry = {"page": i, "verdict": verdict, **sig}
            if best:
                entry.update(best)
            out["pages"].append(entry)
            print(f"  p{i:<3} {verdict:<7} 数字{sig['digit_density']:.2f} "
                  f"短行{sig['short_ratio']:.2f} "
                  + (f"{best['rows']}x{best['cols']}" if best else ""))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只跑文件名含这些词的文件")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    if args.only:
        files = [f for f in files if any(k in f for k in args.only)]

    report = []
    for fn in files:
        path = os.path.join(PDF_DIR, fn)
        print(f"\n{fn}")
        try:
            r = probe(path)
        except Exception as exc:                    # noqa: BLE001
            print(f"  !! {exc}")
            continue
        counts = {}
        for p in r["pages"]:
            counts[p["verdict"]] = counts.get(p["verdict"], 0) + 1
        r["counts"] = counts
        report.append(r)
        print("  小计 " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if args.json:
        json.dump(report, open(args.json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n报告：{args.json}")

    total = {}
    for r in report:
        for k, v in r["counts"].items():
            total[k] = total.get(k, 0) + v
    print("\n=== 全部小计 ===")
    print(" ".join(f"{k}={v}" for k, v in sorted(total.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
