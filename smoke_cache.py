# -*- coding: utf-8 -*-
"""缓存冒烟：同一个问题、两个不同 session，观察第二次是否走缓存。

跨 session 是刻意的——检索只吃 query，与对话历史无关，
所以换了会话也该命中；如果没命中，说明 key 里混进了 session 之类不该有的东西。
"""
import json
import time
import urllib.request
from pathlib import Path

from api_auth import api_key as _api_key

ROOT = Path(__file__).resolve().parent  # 原来写死 /root/rag_deploy，本地跑必挂
KEY = _api_key()
CASES = json.load(open(ROOT / "test_cases2.json", encoding="utf-8"))
Q = CASES[0]["question"]


def ask(session_id):
    body = json.dumps({"message": Q, "session_id": session_id}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/chat/stream",
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": KEY},
    )
    t0 = time.time()
    n = 0
    with urllib.request.urlopen(req, timeout=90) as resp:
        for _ in resp:
            n += 1
    return time.time() - t0, n


print("query:", Q)
t1, n1 = ask("smoke-cache-a")
print("run1 (cold) %.2fs lines=%d" % (t1, n1))
t2, n2 = ask("smoke-cache-b")
print("run2 (warm) %.2fs lines=%d" % (t2, n2))
print("speedup %.2fx" % (t1 / t2 if t2 else 0))
