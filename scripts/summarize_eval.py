#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分类均分 + 失败归因 + 期望文件是否进过来源。"""
import collections
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/eval_law2.json"
rows = json.load(open(path, encoding="utf-8"))["details"]

cat = collections.defaultdict(list)
for r in rows:
    cat[r.get("category") or "?"].append(r)

print("%-10s %6s %6s %6s" % ("类别", "题数", "均分", "召回"))
for k, v in sorted(cat.items(), key=lambda kv: -sum(x.get("score", 0) for x in kv[1]) / len(kv[1])):
    sc = sum(x.get("score", 0) for x in v) / len(v)
    rc = sum(1 for x in v if x.get("recall")) / len(v)
    print("%-10s %6d %6.3f %6.3f" % (k, len(v), sc, rc))

print("\n失败题（得分<1）：")
for r in rows:
    if r.get("score", 0) >= 1:
        continue
    srcs = r.get("sources") or []
    print("  %-5s %-8s %-34s 分=%.2f 归因=%-16s 来源=%s"
          % (r["id"], r.get("category"), r["question"][:32], r["score"],
             r.get("cause"), srcs[:2]))

print("\n归因分布：")
for k, v in collections.Counter(r.get("cause") for r in rows).most_common():
    print("  %-18s %d" % (k, v))
