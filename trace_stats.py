# -*- coding: utf-8 -*-
"""trace_stats.py —— 从 obs.py 落的 JSONL 里统计 P50/P95 与状态分布。

用法：
    # 1) 让后端把 trace 同时落盘（改完下次请求即生效，不用重启）
    echo 'RAG_TRACE_FILE=/tmp/rag_trace.jsonl'  # 写进 .env 或 export 后重启一次

    # 2) 跑一段时间/一轮压测后统计
    python3 trace_stats.py /tmp/rag_trace.jsonl

    # 只看最近 200 条
    python3 trace_stats.py /tmp/rag_trace.jsonl --last 200

为什么要单独一个脚本：`docker logs` 里的 JSON 行不落盘，重启就没了；
而"P95 是多少"这种数字必须能随时复现，不能靠某次截图。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from obs import percentile  # noqa: E402

# 关心的阶段按"用户感知顺序"排，而不是字母序
STAGE_ORDER = ["decide", "graph", "graph_attr", "retrieve", "rerank", "llm", "agent", "finalize"]


def load(path: str, last: int = 0):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # 半行（正在写）直接跳过，不为一条坏数据罢工
            if rec.get("ev") != "req":
                continue
            records.append(rec)
    return records[-last:] if last else records


def report(records):
    if not records:
        print("没有可统计的记录。检查 RAG_TRACE_FILE 是否开启、路径是否正确。")
        return

    print(f"样本数：{len(records)}\n")
    print(f"{'阶段':<12}{'P50(ms)':>10}{'P95(ms)':>10}{'最大':>10}")
    print("-" * 42)

    total = [r.get("total_ms", 0) for r in records]
    print(f"{'总计':<12}{percentile(total, 50):>10}{percentile(total, 95):>10}{max(total):>10.1f}")

    for stage in STAGE_ORDER:
        vals = [r["stages"][stage] for r in records if stage in (r.get("stages") or {})]
        if not vals:
            continue
        print(f"{stage:<12}{percentile(vals, 50):>10}{percentile(vals, 95):>10}{max(vals):>10.1f}")

    ttfts = [r["ttft_ms"] for r in records if r.get("ttft_ms") is not None]
    if ttfts:
        print("-" * 42)
        print(f"{'首字(TTFT)':<12}{percentile(ttfts, 50):>10}{percentile(ttfts, 95):>10}{max(ttfts):>10.1f}")

    # 状态分布：拒答比例突然升高，通常意味着检索门槛或知识库出了问题
    statuses = Counter(r.get("status") for r in records if r.get("status"))
    if statuses:
        print("\n状态分布：")
        for name, n in statuses.most_common():
            print(f"  {name:<14}{n:>4}  {n / len(records):>6.1%}")

    routes = Counter(r.get("route") for r in records if r.get("route"))
    if routes:
        print("\n检索路径：" + "、".join(f"{k} {v}" for k, v in routes.most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="RAG_TRACE_FILE 指向的 JSONL")
    ap.add_argument("--last", type=int, default=0, help="只统计最近 N 条")
    args = ap.parse_args()
    report(load(args.path, args.last))


if __name__ == "__main__":
    main()
