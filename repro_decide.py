# -*- coding: utf-8 -*-
"""repro_decide.py —— 复现「工具决策判成不需要检索」的那一次

背景：三轮评测里 T038「谁住在广州」出现 1 次整题拒答，日志是
  `📝 决策结果：不需要检索` → 没有 graph 阶段 → status=out_of_scope
前两轮同一题正常（route=graph, conf=0.9935）。

要区分两件事：
  A. 这题本身处在决策边界上，失败率高（那 38 题满分有水分）
  B. 全局偶发，1/114（那这是 ReAct 决策的单点故障，需要兜底）

做法：同一题跑 N 次，每次前清缓存（缓存键是问题文本，不清就会命中旧答案）。
"""

import json
import os
import sys
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"


def _api_key() -> str:
    if os.getenv("RAG_API_KEY"):
        return os.environ["RAG_API_KEY"].strip()
    for p in (
        Path(__file__).resolve().parent / ".rag_api_key",  # 本地 WSL 直跑
        "/root/rag_deploy/.rag_api_key",                   # 云端宿主机
        "/app/.rag_api_key",                               # 容器内
    ):
        if os.path.exists(p):
            return open(p, encoding="utf-8").read().strip()
    raise SystemExit("找不到 API key")


H = {"X-API-Key": _api_key()}


def ask(q: str, session: str):
    requests.post(f"{BASE}/admin/cache/clear", headers=H, timeout=10)
    text, final = [], None
    with requests.post(
        f"{BASE}/chat/stream",
        json={"message": q, "session_id": session},
        headers=H, stream=True, timeout=240,
    ) as r:
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            body = raw[6:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except ValueError:
                continue
            if isinstance(obj, dict):
                if isinstance(obj.get("text"), str):
                    text.append(obj["text"])
                if any(k in obj for k in ("status", "confidence", "sources")):
                    final = obj
    final = final or {}
    return "".join(text).strip(), final


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    targets = [
        ("T038", "根据知识库，谁住在广州？", "小陈"),
        ("T001", "根据知识库，小智喜欢什么？", "羊肉串"),
    ]
    for cid, q, must in targets:
        times = n if cid == "T038" else 3
        ok = 0
        fails = []
        print(f"\n=== {cid} × {times} 次：{q}")
        for i in range(1, times + 1):
            ans, final = ask(q, f"repro-{cid}-{i}")
            hit = must in ans
            st = final.get("status")
            ok += 1 if hit else 0
            flag = "OK " if hit else "MISS"
            print(f"  [{i:>2}] {flag} status={st} conf={final.get('confidence')} ans={ans[:46]}")
            if not hit:
                fails.append({"i": i, "status": st, "answer": ans, "final": final})
        print(f"  → 命中 {ok}/{times} = {ok/times:.0%}")
        if fails:
            print(f"  失败样本：{json.dumps(fails[0], ensure_ascii=False)[:400]}")


if __name__ == "__main__":
    main()
