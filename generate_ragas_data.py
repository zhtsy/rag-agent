#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""generate_ragas_data.py —— 通过 HTTP 拉取问答对，生成 RAGAS 评估数据集。

和本地原版的区别
────────────────
原版打的是 `POST /chat`，但这个项目的后端**没有 `/chat`**，只有 `/chat/stream`
（SSE 流式）。直接照搬会 404。这里改成解析 SSE，从终帧取 sources / contexts，
从逐 token 帧拼出 answer。

产出字段严格对齐 RAGAS 要求：
    question / answer / contexts(List[str]) / ground_truth

用法
────
    python3 generate_ragas_data.py                       # 默认 test_cases2+3
    python3 generate_ragas_data.py --cases test_cases3.json --limit 5
    python3 generate_ragas_data.py --out ragas_dataset.json
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = ["test_cases2.json", "test_cases3.json"]

sys.path.insert(0, str(BASE_DIR))
try:
    from api_auth import auth_headers as _auth_headers
except Exception:  # noqa: BLE001
    def _auth_headers() -> dict:
        return {}


def ask_stream(base: str, question: str, session_id: str, timeout: int = 180):
    url = base.rstrip("/") + "/chat/stream"
    body = json.dumps({"message": question, "session_id": session_id}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json", **_auth_headers()}
    )

    chunks, contexts = [], []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if "text" in obj:
                chunks.append(str(obj["text"]))
            if obj.get("contexts"):
                contexts = obj["contexts"]
    return "".join(chunks), contexts


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 RAGAS 评估数据集（HTTP 版）")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    ap.add_argument("--out", default="ragas_dataset.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    cases = []
    for name in args.cases:
        p = Path(name)
        if not p.is_absolute():
            p = BASE_DIR / p
        if not p.exists():
            print(f"⚠️ 跳过不存在的测试集：{p}", file=sys.stderr)
            continue
        cases.extend(json.loads(p.read_text(encoding="utf-8")))
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        print("❌ 没有可用的测试集", file=sys.stderr)
        return 1

    print(f"🚀 共 {len(cases)} 条，开始从 {args.base} 拉取…")
    rows = []
    for i, c in enumerate(cases, 1):
        q = c["question"]
        gt = c.get("expected_answer") or c.get("ground_truth") or ""
        print(f"[{i}/{len(cases)}] {c.get('id', i)} {q}")
        try:
            answer, contexts = ask_stream(args.base, q, f"ragas-{c.get('id', i)}", args.timeout)
        except Exception as e:  # noqa: BLE001 - 单条失败不该中断整个数据集
            print(f"    ❌ {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(args.sleep)
            continue

        if not contexts:
            print("    ⚠️ contexts 为空：RAGAS 的 context_precision/recall 对该条无效")
        rows.append({
            "question": q,
            "answer": answer,
            "contexts": contexts or [""],
            "ground_truth": gt,
        })
        print(f"    len(answer)={len(answer)} contexts={len(contexts)}")
        time.sleep(args.sleep)

    out = Path(args.out)
    if not out.is_absolute():
        out = BASE_DIR / out
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已保存 {len(rows)} 条到 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
