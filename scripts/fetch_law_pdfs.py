# -*- coding: utf-8 -*-
"""从 GitHub 仓库 lawtext/law-flk-vol1-pdf 下载真实法律 PDF 语料。

为什么要这个仓库
────────────────
`knowledge_base_law/` 里 52 部法全是 `.md`，知识库实际上只有一种文件类型。
要验证「PDF/扫描件能不能进库并被检索」，必须用**真实公文 PDF**，而不是自己
用 reportlab 画样本——真实 PDF 的脏版式（每个字独立定位、标点单独成行、
全角数字、双栏公报）是造不出来的。这个仓库是国家法律法规数据库的 PDF 存档，
716 个文件、1.5 GB，和现有 md 同源同领域，天然能做「同一部法、两种格式」对照。

选文件的策略
────────────
1. **与现有 md 重叠的取最新版本**：同一条法在库里可能有 1998/2008/2020 多版，
   只留日期最新的，否则同一条文进库三遍，top-5 名额被同义块吃掉。
2. **按体积分三层各取若干个**：
   - 小于 2 MB —— 基本是文本型 PDF（有文字层），考验抽取质量；
   - 2–6 MB —— 混合（正文 + 内嵌图）；
   - 大于 6 MB —— 基本是扫描件（整页是图），考验 OCR 兜底。
   只下文本型会得出"PDF 没问题"的错觉，只下扫描件又会被 OCR 错误率带偏。
3. **另取一批 md 里没有的法**：如果全部与 md 重叠，而建库又是「md 优先去重」，
   PDF 就一块都进不去，等于什么都没验证。这批是真实增量。

产物
────
    knowledge_base_pdf/<文件名>.pdf
    knowledge_base_pdf/_manifest.json   # 文件名/大小/sha256/来源 URL/分类

重跑是幂等的：已存在且大小一致的文件直接跳过。

用法
────
    python3 scripts/fetch_law_pdfs.py                # 默认 30 重叠 + 10 新增
    python3 scripts/fetch_law_pdfs.py --overlap 10 --fresh 5
    python3 scripts/fetch_law_pdfs.py --list-only    # 只打印选中的文件
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_DIR = os.path.join(ROOT, "knowledge_base_law")
OUT_DIR = os.path.join(ROOT, "knowledge_base_pdf")
API = "https://api.github.com/repos/lawtext/law-flk-vol1-pdf/contents/pdf/%E6%B3%95%E5%BE%8B"
RAW = "https://raw.githubusercontent.com/lawtext/law-flk-vol1-pdf/main/pdf/%E6%B3%95%E5%BE%8B/"

# 2026-11-13 实测：直连反而最快（421 KB/s，0.8 MB 两秒下完），ghfast.top 那次
# 每个连接都挂到 240 s 超时 —— 一个文件能卡 20 分钟。所以改成**直连优先**，
# 并且把单次超时压到 90 s：镜像坏掉时快速换下一个，而不是等满 240 s。
MIRRORS = (
    "",                                  # 直连优先
    "https://ghfast.top/",
    "https://gh-proxy.com/",
)

# 文件名形如 `中华人民共和国专利法_20201017_ff808081752b7d43.pdf`
_NAME_RE = re.compile(r"^(?P<name>.+?)_(?P<date>\d{8})_[0-9a-f]+\.pdf$")

# 体积分档（字节）。分档是为了让"文本型"和"扫描件"都进样本。
SIZE_TIERS = (
    ("text", 0, 2 * 1024**2),
    ("mixed", 2 * 1024**2, 6 * 1024**2),
    ("scan", 6 * 1024**2, 1 << 40),
)


def law_key(name: str) -> str:
    """把「中华人民共和国专利法」压成「专利法」，便于和 md 文件名精确比对。

    **必须精确比对，不能用子串包含**：用 `"刑法" in name` 会把
    「中华人民共和国刑法修正案（十）」和「…关于《刑法》第三十条的解释」
    全部误判成"已有 md"，它们其实是独立文档，不是刑法本体。
    """
    return name.replace("中华人民共和国", "").strip()


# 这些不是"一部法"，是决议/解释/修正案，对企业合规知识库没有检索价值。
# 排除掉，否则采样会挑出一堆「关于植树节的决议」占额度。
_NON_LAW_MARKERS = ("决议", "解释", "修正案", "决定", "办法", "条例", "规定", "批复")


def md_stems() -> list[str]:
    """现有 md 语料的法名（去掉 .md）。`民法典_合同编.md` 这种取主名。"""
    if not os.path.isdir(MD_DIR):
        return []
    stems = []
    for fn in os.listdir(MD_DIR):
        if fn.endswith(".md"):
            stems.append(fn[:-3].split("_")[0])
    return sorted(set(stems))


def fetch_index() -> list[dict]:
    req = urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        data = json.load(r)
    if not isinstance(data, list):
        raise SystemExit(f"GitHub API 返回异常：{data}")
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
        })
    return out


def pick_latest(items: list[dict]) -> list[dict]:
    """同一条法多版本 → 只留日期最新的一版。"""
    best: dict[str, dict] = {}
    for it in items:
        cur = best.get(it["law"])
        if cur is None or it["date"] > cur["date"]:
            best[it["law"]] = it
    return list(best.values())


def tier_of(size: int) -> str:
    for name, lo, hi in SIZE_TIERS:
        if lo <= size < hi:
            return name
    return "scan"


def choose(cands: list[dict], overlap: int, fresh: int, md: list[str]) -> list[dict]:
    """选出要下载的文件。

    `overlap` 是"和现有 md 重叠"的配额，`fresh` 是"md 没有的新法"配额。
    两边都按体积分层轮转，保证每一档都有代表，而不是被某一档吃光名额
    （Unicode 排序会把同档文件全排在前面，之前采样就吃过这个亏）。
    """
    def overlaps(item):
        return law_key(item["law"]) in md_set

    md_set = set(md)
    pool_over = pick_latest([c for c in cands if overlaps(c)])
    # 「新增」只收正规法律（…法），把决议/解释/修正案挡在门外
    pool_fresh = pick_latest([
        c for c in cands
        if not overlaps(c)
        and c["law"].endswith("法")
        and not any(m in c["law"] for m in _NON_LAW_MARKERS)
    ])

    def layered(pool, quota):
        by: dict[str, list[dict]] = {}
        for it in pool:
            by.setdefault(tier_of(it["size"]), []).append(it)
        for k in by:
            by[k].sort(key=lambda x: x["size"])
        order = ["text", "scan", "mixed"]     # 先各档取一个，再轮转
        picked, i = [], 0
        while len(picked) < quota and any(by.values()):
            k = order[i % len(order)]
            if by.get(k):
                picked.append(by[k].pop(0))
            i += 1
            if i > quota * 6:
                break
        return picked

    return layered(pool_over, overlap) + layered(pool_fresh, fresh)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(item: dict, retries: int = 2, timeout: float = 90.0) -> str:
    dst = os.path.join(OUT_DIR, item["file"])
    if os.path.exists(dst) and os.path.getsize(dst) == item["size"]:
        return "skip"
    url = urllib.parse.quote(item["url"], safe=":/?&=%")
    last: Exception | None = None
    for mirror in MIRRORS:
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(mirror + url,
                                             headers={"User-Agent": "Mozilla/5.0"})
                # 扫描件动辄 20 MB，给足超时；但镜像挂住时不能一直等，
                # 90 s 足够 200 KB/s 下完 18 MB 的九成，超了就说明这条链路坏了。
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
                if len(data) != item["size"]:
                    raise IOError(f"大小不符：{len(data)} != {item['size']}")
                with open(dst, "wb") as f:
                    f.write(data)
                return f"ok({mirror or 'direct'})"
            except Exception as exc:      # noqa: BLE001 换镜像继续试
                last = exc
                time.sleep(attempt)
    raise SystemExit(f"下载失败 {item['file']}: {last}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=int, default=30, help="与现有 md 重叠的文件数")
    ap.add_argument("--fresh", type=int, default=10, help="md 里没有的新法文件数")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args(argv)

    md = md_stems()
    cands = fetch_index()
    chosen = choose(cands, args.overlap, args.fresh, md)

    md_set = set(md)
    print(f"仓库文件 {len(cands)} 个 ｜ 现有 md {len(md)} 部 ｜ 选中 {len(chosen)} 个")
    for it in chosen:
        tag = "重叠" if law_key(it["law"]) in md_set else "新增"
        print(f"  [{tag}/{tier_of(it['size'])}] {it['file']}  {it['size']/2**20:.1f} MB")
    if args.list_only:
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            manifest = json.load(open(manifest_path, encoding="utf-8"))
        except Exception:
            manifest = {}

    t0 = time.time()
    for i, it in enumerate(chosen, 1):
        status = download(it)
        path = os.path.join(OUT_DIR, it["file"])
        manifest[it["file"]] = {
            "law": it["law"],
            "date": it["date"],
            "size": it["size"],
            "tier": tier_of(it["size"]),
            "overlap_md": law_key(it["law"]) in md_set,
            "sha256": sha256_of(path),
            "url": it["url"],
        }
        print(f"[{i}/{len(chosen)}] {status} {it['file']} ({it['size']/2**20:.1f} MB)")
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    total = sum(v["size"] for v in manifest.values())
    print(f"\n完成：{len(manifest)} 个文件，{total/2**20:.1f} MB，耗时 {time.time()-t0:.0f}s")
    print(f"清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
