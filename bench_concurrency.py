# -*- coding: utf-8 -*-
"""bench_concurrency.py —— 并发压测：不同并发下的 TTFT 与总耗时分布。

为什么需要它
────────────
"2 vCPU 能扛多少并发"是这套系统最容易被问、也最容易答错的问题。
答错的典型方式是拍一个"理论 QPS"；正确答案只能压出来，而且要分清两个指标：

  · TTFT（首字延迟）：用户感知的"快不快"，流式系统里比总耗时更重要
  · 总耗时：决定服务端吞吐和排队长度

两者在 CPU 密集的 rerank 上会同时恶化，但恶化速度不一样——这正是要看曲线的理由。

用法
────
    python3 bench_concurrency.py --concurrency 1,2,4 --rounds 6
    python3 bench_concurrency.py --url http://127.0.0.1:8000 --concurrency 1,4 --rounds 8

输出：一份 JSON（明细）+ 一张 Markdown 表（可直接贴进 README）。

注意
────
这是**打自己的服务**，不是打公网；跑之前确认没有真人正在用。
并发别一上来就 16：这台机器 rerank 是 CPU 串行的，16 并发会让单请求 TTFT
涨到分钟级，压测脚本自己就先超时了，测不出有用信息。
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obs import percentile  # noqa: E402

# 用不同的问题轮换，避免总压同一条（真实场景里问题分布是散的）
QUESTIONS = [
    "根据知识库，小陈养了什么宠物？",
    "根据知识库，小智喜欢什么？",
    "根据知识库，小张住在哪里？",
    "根据知识库，小陈的宠物叫什么品种？",
    "根据知识库，小李的工作是什么？",
    "根据知识库，马斯克是谁？",          # 越界题，走"检索落空"这条路径
]


def _api_key() -> str:
    key = os.environ.get("RAG_API_KEY", "").strip()
    if key:
        return key
    for path in ("/app/.rag_api_key", "./.rag_api_key", "/root/rag_deploy/.rag_api_key"):
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return ""


def one_request(url: str, question: str, api_key: str, timeout: float) -> dict:
    """发一条流式请求，记录 TTFT / 总耗时 / 收尾元信息。"""
    started = time.perf_counter()
    first_at = None
    status = None
    conf = None
    timings = None
    n_events = 0

    try:
        with requests.post(
            f"{url}/chat/stream",
            json={"message": question, "session_id": f"bench-{time.time_ns()}"},
            headers={"X-API-Key": api_key, "Accept": "text/event-stream"},
            stream=True,
            timeout=timeout,
        ) as resp:
            if resp.status_code != 200:
                return {"ok": False, "http": resp.status_code, "q": question,
                        "total_ms": round((time.perf_counter() - started) * 1000, 1)}

            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                payload = raw[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                n_events += 1
                if "text" in data and first_at is None:
                    first_at = time.perf_counter()      # 第一个正文 token 到手
                if "sources" in data:
                    status = data.get("status")
                    conf = data.get("confidence")
                    timings = data.get("timings")

    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": type(e).__name__, "q": question,
                "total_ms": round((time.perf_counter() - started) * 1000, 1)}

    return {
        "ok": True,
        "q": question,
        "ttft_ms": round((first_at - started) * 1000, 1) if first_at else None,
        "total_ms": round((time.perf_counter() - started) * 1000, 1),
        "status": status,
        "conf": conf,
        "events": n_events,
        "server_timings": timings,
    }


def run_level(url: str, api_key: str, concurrency: int, rounds: int, timeout: float) -> dict:
    """在给定并发度下跑 rounds 条请求，返回统计结果。"""
    answers = []
    began = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(one_request, url, QUESTIONS[i % len(QUESTIONS)], api_key, timeout)
            for i in range(rounds)
        ]
        for f in futures:
            answers.append(f.result())

    wall = time.perf_counter() - began
    ok = [a for a in answers if a.get("ok")]
    ttfts = [a["ttft_ms"] for a in ok if a.get("ttft_ms") is not None]
    totals = [a["total_ms"] for a in ok]

    return {
        "concurrency": concurrency,
        "requests": rounds,
        "ok": len(ok),
        "failed": len(answers) - len(ok),
        "wall_s": round(wall, 1),
        # 吞吐按"整轮墙钟时间"算，包含排队，才是用户实际体验到的吞吐
        "rps": round(len(ok) / wall, 2) if wall > 0 else None,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "total_p50": percentile(totals, 50),
        "total_p95": percentile(totals, 95),
        "details": answers,
    }


def markdown_table(levels) -> str:
    lines = [
        "| 并发 | 请求数 | 成功率 | 吞吐(req/s) | TTFT P50 | TTFT P95 | 总耗时 P50 | 总耗时 P95 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lv in levels:
        rate = f"{lv['ok'] / lv['requests']:.0%}" if lv["requests"] else "-"
        lines.append(
            f"| {lv['concurrency']} | {lv['requests']} | {rate} | {lv['rps']} "
            f"| {lv['ttft_p50']} ms | {lv['ttft_p95']} ms "
            f"| {lv['total_p50']} ms | {lv['total_p95']} ms |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", default="1,2,4", help="逗号分隔，如 1,2,4")
    ap.add_argument("--rounds", type=int, default=6, help="每个并发度发多少条请求")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="bench_result.json")
    args = ap.parse_args()

    api_key = _api_key()
    if not api_key:
        print("找不到 API Key：设 RAG_API_KEY 环境变量，或确保 .rag_api_key 存在")
        return 2

    levels = []
    for level in [int(x) for x in args.concurrency.split(",") if x.strip()]:
        print(f"▶ 并发 {level}：{args.rounds} 条请求…", flush=True)
        result = run_level(args.url, api_key, level, args.rounds, args.timeout)
        levels.append(result)
        print(f"   TTFT P50 {result['ttft_p50']} ms / P95 {result['ttft_p95']} ms，"
              f"吞吐 {result['rps']} req/s，失败 {result['failed']}", flush=True)

    table = markdown_table(levels)
    print("\n" + table + "\n")

    Path(args.out).write_text(
        json.dumps({"levels": levels, "table": table}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"明细已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
