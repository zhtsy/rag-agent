# -*- coding: utf-8 -*-
"""依赖降级演练：Neo4j 停掉之后，问答链路是 500、是卡死、还是降级。

只看"服务还在不在"没有意义——要看三件事：
  1. HTTP 状态码（500 就是硬故障，200 才谈得上降级）
  2. 耗时（有没有卡在连接超时上，那比直接报错更糟）
  3. 回答内容（证据够不够，有没有变成"无法回答"）

用两个问题：一个走图谱（关系类），一个走向量（普通事实）。
这样能看出"只有图谱挂了"时，两类问题是否都还能答。
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from api_auth import api_key as _api_key

ROOT = Path(__file__).resolve().parent  # 原来写死 /root/rag_deploy，本地跑必挂
KEY = _api_key()
URL = "http://127.0.0.1:8000/chat/stream"

# 用例文件和题号都可传，因为"停掉依赖再问同一个问题"会被上一轮的缓存直接命中，
# 那样测到的不是降级，是缓存。每次都要换没问过的问题。
CASES_FILE = sys.argv[2] if len(sys.argv) > 2 else "test_cases2.json"
PICK = (sys.argv[3] if len(sys.argv) > 3 else "T037,T001").split(",")
CASES = {
    c["id"]: c
    for c in json.load(open(ROOT / CASES_FILE, encoding="utf-8"))
}


def ask(question, timeout=150):
    body = json.dumps({"message": question, "session_id": "drill"}).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json", "X-API-Key": KEY}
    )
    t0 = time.time()
    answer, status = [], None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "text" in obj:
                    answer.append(obj["text"])
                elif obj.get("error"):
                    answer.append(f"[error] {obj['error']}")
    except urllib.error.HTTPError as exc:
        status = exc.code
        answer.append(f"[HTTPError {exc.code}] {exc.reason}")
    except Exception as exc:  # 超时/连接重置都属于"比报错更糟"的一类
        status = type(exc).__name__
        answer.append(f"[{type(exc).__name__}] {exc}")
    return status, round(time.time() - t0, 1), "".join(answer).replace("\n", " ")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    rows = []
    for cid in PICK:
        q = CASES[cid]["question"]
        status, dt, ans = ask(q)
        rows.append({"id": cid, "status": status, "elapsed": dt, "answer": ans})
        print(f"[{label}] {cid} status={status} {dt}s")
        print(f"    A: {ans[:200]}")
    out = f"/tmp/drill_{label}.json"
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
