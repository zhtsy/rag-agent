#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查最近一次 CI 的状态与失败步骤（不依赖 gh）。"""
import json
import urllib.request

REPO = "zhtsy/rag-agent"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ci-check"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


runs = get(f"https://api.github.com/repos/{REPO}/actions/runs?per_page=1")
run = runs["workflow_runs"][0]
print(run["head_sha"][:7], run["status"], run["conclusion"])
jobs = get(f"https://api.github.com/repos/{REPO}/actions/runs/{run['id']}/jobs")
for j in jobs["jobs"]:
    print(f"\n[{j['name']}] {j['conclusion']}")
    for s in j.get("steps") or []:
        mark = "✓" if s.get("conclusion") == "success" else ("-" if s.get("conclusion") == "skipped" else "✗")
        print(f"  {mark} {s['name']}  ({s.get('conclusion')})")
