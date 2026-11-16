#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对已有消融报告里的全部答案跑一遍 hedged_guess，人工核对是否误判。

为什么先离线跑：四组报告里已经有 154 份真实答案，比重新打后端快，
而且覆盖了"表格块被删"这种线上不会出现的极端条件。
"""
import json
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR.parent.parent))

from confidence import hedged_guess  # noqa: E402

total = 0
hits = []
for tag in ("A", "B", "C", "D"):
    p = DIR / f"{tag}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    for r in d["details"]:
        total += 1
        if hedged_guess(r.get("answer", "")):
            hits.append((tag, r["id"], r["score"], r["question"], r["answer"]))

print(f"共 {total} 份答案，hedged 命中 {len(hits)} 条\n")
for tag, qid, score, q, a in hits:
    print(f"[{tag}] {qid} 得分 {score} ｜ {q}")
    print(f"    答：{a[:200].replace(chr(10), ' ')}")
    print()
