# -*- coding: utf-8 -*-
"""补一批**带附表**的法规 PDF —— 税率表、收费表、限额表。

为什么单独补
────────────
现有 `knowledge_base_pdf/` 的 40 个文件全在「法律」目录，都是条款正文：
上一次做「让位体检」时统计过，`附表` 在整份语料里出现 **0 次**，表格块 **0 个**。
也就是说知识库里根本没有表格型内容，检索链路里「表格怎么切、怎么检索」这一
环节从来没被真实数据考验过 —— 评测分好看，是因为题目里没有表格题。

这批文件挑的都是**正文末尾或附件页带表格**的：税法附《税目税额表》，
《诉讼费用交纳办法》附案件受理费标准表。它们的表格页是"真表格"，
不是条款里排比出来的数字。

选文件的约束
────────────
1. 优先挑 md 里没有的法（资源税法、车船税法、印花税法…），这样不会被
   「md 优先让位」逻辑挡掉，一定进库，是真实增量。
2. 只挑小体积（≤6 MB）。附表页数少，几十 MB 的扫描件性价比低，
   而且镜像带宽只有约 200 KB/s。
3. 同一条法多版本取最新（`_pick_latest`），否则同一张税率表进库三遍。

产物
────
    knowledge_base_pdf/<文件名>.pdf
    knowledge_base_pdf/_manifest.json  # 追加 kind="table" 与表格页探测结果

用法
────
    python3 scripts/fetch_table_pdfs.py --list-only
    python3 scripts/fetch_table_pdfs.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fetch_law_pdfs import (  # noqa: E402 复用镜像与校验逻辑
    _NAME_RE,
    download,
    law_key,
    md_stems,
    sha256_of,
    tier_of,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "knowledge_base_pdf")
MANIFEST = os.path.join(OUT_DIR, "_manifest.json")
MIRRORS = None  # download() 内部已按 MIRRORS 轮询

# 要补的法名（不含日期与 hash）。命中的候选再按日期取最新。
TARGET_LAWS = [
    # ── 税法：附件里是《税目税额表》──
    "中华人民共和国资源税法",
    "中华人民共和国车船税法",
    "中华人民共和国耕地占用税法",
    "中华人民共和国印花税法",
    "中华人民共和国契税法",
    "中华人民共和国环境保护税法",
    "中华人民共和国车辆购置税法",
    "中华人民共和国城市维护建设税法",
    # ── 条例：收费表 / 限额表 ──
    "中华人民共和国土地增值税暂行条例",
    "仲裁委员会仲裁收费办法",
    "诉讼费用交纳办法",
    "中华人民共和国港口间海上旅客运输赔偿责任限额规定",
]

MAX_SIZE = 8 * 1024**2   # 超过 8 MB 的不下（带宽约 200 KB/s）


def _index(dirname: str) -> list[dict]:
    url = ("https://api.github.com/repos/lawtext/law-flk-vol1-pdf/contents/pdf/"
           + urllib.parse.quote(dirname))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.load(r)
    out = []
    for it in data:
        m = _NAME_RE.match(it["name"])
        if not m:
            continue
        out.append({
            "file": it["name"],
            "law": m.group("name"),
            "date": m.group("date"),
            "size": it.get("size", 0),
            "url": it["download_url"],
            "dir": dirname,
        })
    return out


def pick(cands: list[dict], want: list[str]) -> list[dict]:
    """按法名精确匹配，同一条法只留最新一版，跳过超大文件。"""
    by: dict[str, list[dict]] = {}
    want_set = set(want)
    for c in cands:
        if c["law"] in want_set and c["size"] <= MAX_SIZE:
            by.setdefault(c["law"], []).append(c)
    picked = []
    for name in want:
        group = by.get(name)
        if not group:
            print(f"  !! 未找到（或超体积）：{name}")
            continue
        picked.append(max(group, key=lambda x: x["date"]))
    return picked


def probe_table_pages(path: str) -> dict:
    """粗测这份 PDF 里有没有"表格页"。

    只看两件事，都便宜：
      · 每页文字里的**数字密度**（数字字符 / 全部字符）。税率表、赔偿限额表
        满页都是 0.5%、3%、2 元，密度远高于条款正文；
      · 每页的**短行比例**（<=12 字的行占比）。表格被抽成文本后是一串短行，
        条款正文是 40 字以上的长句。

    两个都高才算候选。宁可漏判不可误判：漏判只是少标一页，误判会让
    「表格切分」逻辑作用在正文上，把整段切碎。
    """
    try:
        import pypdf
    except ImportError:
        return {"table_pages": [], "note": "no pypdf"}

    try:
        reader = pypdf.PdfReader(path)
    except Exception as exc:                        # noqa: BLE001 坏文件不该中断整批
        return {"table_pages": [], "note": f"read-error: {exc}"}

    hits, empty, total = [], 0, 0
    for i, page in enumerate(reader.pages, 1):
        total += 1
        try:
            text = page.extract_text() or ""
        except Exception:                           # noqa: BLE001
            text = ""
        if len(text.strip()) < 20:
            empty += 1
            continue
        digits = sum(ch.isdigit() for ch in text)
        density = digits / max(1, len(text))
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        short = sum(1 for ln in lines if len(ln) <= 12)
        short_ratio = short / max(1, len(lines))
        if density >= 0.12 and short_ratio >= 0.5:
            hits.append({"page": i, "digit_density": round(density, 3),
                         "short_ratio": round(short_ratio, 3)})
    return {"table_pages": hits, "pages": total, "empty_pages": empty}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args(argv)

    cands = _index("法律") + _index("行政法规")
    chosen = pick(cands, TARGET_LAWS)
    md = set(md_stems())

    print(f"候选池 {len(cands)} 个 ｜ 选中 {len(chosen)} 个")
    for it in chosen:
        tag = "重叠" if law_key(it["law"]) in md else "新增"
        print(f"  [{tag}/{tier_of(it['size'])}] {it['file']}  {it['size']/2**20:.1f} MB")
    if args.list_only:
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    if os.path.exists(MANIFEST):
        try:
            manifest = json.load(open(MANIFEST, encoding="utf-8"))
        except Exception:                           # noqa: BLE001
            manifest = {}

    for i, it in enumerate(chosen, 1):
        try:
            status = download(it)
        except SystemExit as exc:
            print(f"[{i}/{len(chosen)}] 失败 {it['file']}: {exc}")
            continue
        path = os.path.join(OUT_DIR, it["file"])
        probe = probe_table_pages(path)
        manifest[it["file"]] = {
            "law": it["law"],
            "date": it["date"],
            "size": it["size"],
            "tier": tier_of(it["size"]),
            "kind": "table",
            "overlap_md": law_key(it["law"]) in md,
            "sha256": sha256_of(path),
            "url": it["url"],
            "probe": probe,
        }
        json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        n_tbl = len(probe.get("table_pages", []))
        print(f"[{i}/{len(chosen)}] {status} {it['file']} ({it['size']/2**20:.1f} MB) "
              f"页数={probe.get('pages')} 空页={probe.get('empty_pages')} 疑似表格页={n_tbl}")
        for h in probe.get("table_pages", [])[:6]:
            print(f"      p{h['page']} 数字密度={h['digit_density']} 短行={h['short_ratio']}")

    print(f"\n清单：{MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
