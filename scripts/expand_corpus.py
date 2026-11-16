"""从本机 Laws 仓库扩充 md 语料（司法解释 / 行政法规 / 案例）。

为什么要这一步：现有 53 部 md 只覆盖 73 万字，与"法律问答"这个任务的实际
提问面差得太远——实务里被问得最多的往往是**司法解释**（最高法对法条的适用
解释），而语料里一部都没有。

选取策略：按文件体积降序取，**大的通常是有完整条文的实质规范**，小的多是
批复/函（一页纸）。这不是精确的相关性判断，是"用有限预算覆盖最多正文"的
工程取舍，取舍依据会打印出来。

用法：
    python scripts/expand_corpus.py --dry-run
    python scripts/expand_corpus.py --司法解释 300 --行政法规 120
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

RAW = "/home/zh/data/laws_raw/Laws-master"
DST = "knowledge_base_law"
GROUP_PREFIX = {"案例": "案例_"}


def collect(group: str, limit: int | None):
    root = os.path.join(RAW, group)
    if not os.path.isdir(root):
        return []
    items = []
    for cur, dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".md"):
                continue
            p = os.path.join(cur, fn)
            items.append((os.path.getsize(p), p, fn))
    items.sort(key=lambda x: -x[0])
    return items[:limit] if limit else items


def stem_of(fn: str) -> str:
    s = os.path.splitext(fn)[0]
    # 去掉文件名里的日期/编号后缀：`xxx_20210101`、`xxx(2021修正)`
    s = re.sub(r"[（(]\d{4}[^）)]*[）)]$", "", s)
    return s.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--司法解释", dest="sifa", type=int, default=300)
    ap.add_argument("--行政法规", dest="xingzheng", type=int, default=120)
    ap.add_argument("--案例", dest="anli", type=int, default=57)
    args = ap.parse_args()

    existing = {os.path.splitext(f)[0] for f in os.listdir(DST) if f.endswith(".md")}
    plan = []
    for group, limit in (("司法解释", args.sifa), ("行政法规", args.xingzheng),
                         ("案例", args.anli)):
        picked = collect(group, limit)
        chars = sum(p[0] for p in picked) // 2
        print(f"{group}: 选 {len(picked)} 篇，约 {chars/10000:.1f} 万字，"
              f"最小 {picked[-1][0]//2 if picked else 0} 字")
        for size, path, fn in picked:
            stem = GROUP_PREFIX.get(group, "") + stem_of(fn)
            if stem in existing:
                continue
            existing.add(stem)
            plan.append((path, stem))

    total = sum(os.path.getsize(p) for p, _ in plan) // 2
    print(f"\n合计新增 {len(plan)} 篇，约 {total/10000:.1f} 万字")
    if args.dry_run:
        for p, s in plan[:15]:
            print("  ", s)
        print("   ...")
        return 0

    os.makedirs(DST, exist_ok=True)
    for path, stem in plan:
        shutil.copyfile(path, os.path.join(DST, stem + ".md"))
    print(f"✅ 已写入 {DST}: {len(plan)} 个文件")
    print(f"   目录现有 md: {len([f for f in os.listdir(DST) if f.endswith('.md')])} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
