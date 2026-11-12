#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用例集自检：id 唯一、关键词不撞车、每条 must 都能在源文件里找到。"""
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = os.path.join(ROOT, "test_cases_law.json")
DIR = os.path.join(ROOT, "knowledge_base_law")

cases = json.load(open(CASES, encoding="utf-8"))
print("总数", len(cases))
for cat, n in collections.Counter(c["category"] for c in cases).most_common():
    print("  %-8s %d" % (cat, n))

ids = [c["id"] for c in cases]
dup_id = [k for k, v in collections.Counter(ids).items() if v > 1]
print("重复 id:", dup_id or "无")

# 1) 关键词撞车
seen, collide = {}, []
for c in cases:
    if c["category"] == "拒答":
        continue
    for kw in c["must_contain"]:
        if kw in seen:
            collide.append((c["id"], seen[kw], kw))
        seen[kw] = c["id"]
print("关键词撞车:", collide or "无")

# 2) must 是否真的在源文件里（金标可核验）
miss = []
for c in cases:
    f = c.get("expected_source") or ""
    if not f or isinstance(f, list):
        continue
    p = os.path.join(DIR, f)
    if not os.path.exists(p):
        p = os.path.join(ROOT, "knowledge_base", f)
    if not os.path.exists(p):
        miss.append((c["id"], "源文件缺失", f))
        continue
    text = open(p, encoding="utf-8", errors="ignore").read()
    flat = "".join(ch for ch in text if ch not in " \n\t　，。、；：？！“”‘’（）《》,.;:?!\"'()[]-")
    for kw in c["must_contain"]:
        if kw.lower() not in flat.lower():
            miss.append((c["id"], kw, f))
print("关键词不在源文件里:", miss or "无")

# 3) 覆盖了多少部法律
laws = {c["expected_source"] for c in cases if isinstance(c.get("expected_source"), str)}
print("涉及源文件数:", len([x for x in laws if x]))

print("\n主题定位 / 真实业务题：")
for c in cases:
    if c["category"] == "主题定位" or c["id"].startswith("B"):
        print("  %-6s %-8s %s" % (c["id"], c["category"], c["question"][:52]))
