#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""看回归报告的 status 分布，确认 hedged 没有误标。

报告里没有 status 字段时（本轮改动之前生成的），本地用 confidence 的纯函数重算
——三个输入（答案文本、有没有来源）报告里都有，所以重算等价于重跑，但不花时间。
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from confidence import answer_status, hedged_guess, looks_like_abstention  # noqa: E402

pattern = sys.argv[1] if len(sys.argv) > 1 else "law_regress_*.json"
reports = sorted((ROOT / "reports").glob(pattern))
if not reports:
    print(f"没找到报告：reports/{pattern}")
    raise SystemExit(1)

p = reports[-1]
d = json.loads(p.read_text(encoding="utf-8"))
print(f"报告：{p.name}")
print(f"条数 {d['total']} ｜ 均分 {d['avg_score']} ｜ 文件 {d.get('case_files')}")

cnt, hedged_rows, recomputed = Counter(), [], 0
for r in d["details"]:
    if "error" in r:
        continue
    ans = r.get("answer") or ""
    st = r.get("status")
    if not st:
        st = answer_status(looks_like_abstention(ans), bool(r.get("sources")),
                           hedged_guess(ans))
        recomputed += 1
    cnt[st] += 1
    if st == "hedged":
        hedged_rows.append(r)

if recomputed:
    print(f"（{recomputed} 条报告无 status，已用纯函数重算）")
print("status 分布：", dict(cnt))
print(f"hedged 条数：{len(hedged_rows)}")
for r in hedged_rows:
    print(f"  {r['id']} {r['question'][:30]} 得分 {r['score']}")

# missed = 有来源却被判成拒答。要么是真漏答，要么是拒答判定误判，两种都得看原文。
missed = [r for r in d["details"]
          if "error" not in r and (r.get("status") or
                                   answer_status(looks_like_abstention(r.get("answer") or ""),
                                                 bool(r.get("sources")),
                                                 hedged_guess(r.get("answer") or ""))) == "missed"]
print(f"\nmissed 条数：{len(missed)}（有来源却判成拒答，需人工看原文）")
for r in missed:
    print(f"  {r['id']} 得分 {r['score']} sources={r.get('sources')}")
    print(f"    {(r.get('answer') or '')[:160].replace(chr(10), ' ')}")
