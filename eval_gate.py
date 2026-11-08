#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测门禁：跑全量评测，不达标就 exit 1。

为什么不做进 GitHub Actions
---------------------------
端到端评测要起 uvicorn + Chroma + Neo4j + MySQL + 两个 bge 模型（约 1.2GB 内存）。
GitHub runner 上没有这些，硬塞进去只有两种结果：要么长期红色（假门禁），
要么把评测改成不依赖模型（那就不是端到端了）。
所以分工是：**CI 跑纯逻辑单测，这道门放在有模型的地方，每天跑一次。**

判定三条，任一不满足就失败：
  1. 均分 >= --min-score
  2. 成功条数 == 总条数（有失败说明接口或依赖挂了，分数再高也不算过）
  3. 跑之前 /health 必须正常

关于 --repeat
-------------
单次跑出来的 0.9474 是个点估计，不是真值。同一套用例换个时间再跑，
模型采样和检索路径都可能变，分数会抖（历史上见过 36/38 与 35/38）。
只报最好的那次等于自欺，所以我加了 --repeat：跑 N 轮，报均值与极差。

⚠️ 重复时必须清缓存。否则第 2、3 轮全部命中缓存，拿到的是同一份答案，
方差恒为 0 —— 测出来的"稳定"是缓存造成的假象，不是模型的稳定性。
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# 项目根目录用本文件位置推导：云端是 /root/rag_deploy，本地是 ~/py_code。
# 原来这里写死 "/root/rag_deploy"，迁移到本地后连密钥文件都读不到，
# 表现是"闸门一跑就 WARN 清缓存失败 / 401"，而服务其实是好的。
APP = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8000"
HEALTH_URL = f"{BASE}/health"
CLEAR_URL = f"{BASE}/admin/cache/clear"


def read_health():
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
            return json.load(resp)
    except Exception as exc:
        return {"error": str(exc)}


def clear_cache():
    """清空后端检索缓存。失败不算致命（单轮评测本来就是冷启动），但要说出来。"""
    try:
        key = (APP / ".rag_api_key").read_text(encoding="utf-8").strip()
        req = urllib.request.Request(
            CLEAR_URL, data=b"", method="POST", headers={"X-API-Key": key}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except Exception as exc:
        print(f"WARN 清缓存失败（{exc}）；本轮可能命中上一轮的缓存，方差会失真")
        return None


def run_once(cases: str, out_dir: Path, stamp: str, idx: int):
    report = out_dir / f"gate_{stamp}_r{idx}.json"
    log = out_dir / f"gate_{stamp}_r{idx}.log"
    cmd = [sys.executable, "eval_system.py", "--cases", cases, "--out", str(report)]
    started = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(APP), stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - started

    if rc != 0 or not report.exists():
        return {"ok": False, "rc": rc, "log": str(log), "elapsed": elapsed}

    data = json.loads(report.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "report": str(report),
        "log": str(log),
        "score": float(data.get("avg_score") or 0),
        "total": int(data.get("total") or 0),
        "success": int(data.get("success") or 0),
        "elapsed": elapsed,
        "per_topic": data.get("by_topic") or data.get("topics") or {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="test_cases2.json")
    parser.add_argument("--min-score", type=float, default=0.90)
    parser.add_argument("--out", default="/root/logs")
    parser.add_argument("--repeat", type=int, default=1,
                        help="重复跑 N 轮，报均值与极差（每轮前清缓存）")
    args = parser.parse_args()

    if args.repeat < 1:
        print("FAIL --repeat 必须 >= 1")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    health = read_health()
    if health.get("error") or health.get("status") != "ok":
        print(f"FAIL 服务不健康，先修服务再谈分数: {health}")
        return 1

    # 缓存开着时，耗时不能代表冷路径。分数仍然可信（答案一样），
    # 但不能拿这份报告的耗时和历史基线比。打印出来，别让人事后猜。
    cache = health.get("cache", {})
    if cache.get("enabled") and args.repeat == 1:
        print(f"NOTE 检索缓存已开启（hits={cache.get('hits')}），耗时指标仅供参考")

    rounds = []
    for i in range(1, args.repeat + 1):
        if args.repeat > 1:
            clear_cache()
        print(f"== 第 {i}/{args.repeat} 轮 ==")
        r = run_once(args.cases, out_dir, stamp, i)
        rounds.append(r)
        if not r["ok"]:
            print(f"FAIL 评测进程异常 rc={r['rc']}，日志 {r['log']}")
            return 1
        print(f"   均分 {r['score']:.4f} ｜ 成功 {r['success']}/{r['total']} ｜ 耗时 {r['elapsed']:.0f}s")

    scores = [r["score"] for r in rounds]
    mean = statistics.fmean(scores)
    lo, hi = min(scores), max(scores)
    spread = hi - lo
    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0

    summary = {
        "cases": args.cases,
        "repeat": args.repeat,
        "scores": scores,
        "mean": round(mean, 6),
        "min": lo,
        "max": hi,
        "spread": round(spread, 6),
        "stdev": round(stdev, 6),
        "min_score_threshold": args.min_score,
        "rounds": rounds,
    }
    summary_path = out_dir / f"gate_{stamp}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 56}")
    if args.repeat > 1:
        print(f"分数分布 {['%.4f' % s for s in scores]}")
        print(f"均值 {mean:.4f} ｜ 极差 {spread:.4f} ｜ 标准差 {stdev:.4f}")
        # 极差超过 3 个百分点，说明结论已经不稳，别只报均值
        if spread > 0.03:
            print(f"WARN 单轮间极差 {spread:.4f} > 0.03：这个分数段不足以做细粒度决策")
    else:
        print(f"均分 {mean:.4f}")
    print(f"汇总 {summary_path}")

    failures = []
    if any(r["total"] == 0 for r in rounds):
        failures.append("总条数为 0，用例文件可能读错了")
    bad = [r for r in rounds if r["success"] != r["total"]]
    if bad:
        failures.append(
            "有轮次存在请求失败：" + ", ".join(f"{r['success']}/{r['total']}" for r in bad)
        )
    if mean < args.min_score:
        failures.append(f"均值 {mean:.4f} < 门槛 {args.min_score:.4f}")

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
