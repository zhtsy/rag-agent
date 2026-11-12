#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""看对照实验里哪几题失分，以及两臂的答案差在哪。"""

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ab_fullcontext.json"
d = json.load(open(path, encoding="utf-8"))
print("题数:", d["summary"]["n"])

for r in d["rows"]:
    a = r.get("a_score", 1.0)
    b = r.get("b_score", 1.0)
    if a < 1.0 or b < 1.0:
        print("\n" + r.get("id", "?"), "A=", a, "B=", b)
        print("  Q:", r["question"][:70])
        print("  A答:", (r.get("a_answer") or "")[:110])
        print("  B答:", (r.get("b_answer") or "")[:110])
        print("  上下文: A=", r.get("a_ctx_chars"), "字  B=", r.get("b_ctx_chars"), "字")

s = d["summary"]
print("\n=== 汇总 ===")
print("A(全塞) 得分", s["a_score"], "耗时", s["a_sec"], "s  上下文", s["a_ctx_chars"], "字")
print("B(检索) 得分", s["b_score"], "耗时", s["b_sec"], "s  上下文", s["b_ctx_chars"], "字")
if s.get("b_ctx_chars"):
    print("上下文压缩比:", round(s["a_ctx_chars"] / s["b_ctx_chars"], 1), "x")
