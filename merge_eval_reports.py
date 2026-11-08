#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_eval_reports.py —— 合并分批的评测报告，并重算分层归因。

为什么要这个
────────────
38 条的完整评测要跑 7 分钟以上，而远程终端（workbench exec）单次命令会被
超时打断——打断后报告文件往往还没写出来。所以一直是用 `--limit/--offset`
分两批跑。但**归因必须在全量上算**：单看一半，"丢在哪一层"的比例会失真。

本脚本把若干份报告拼成一份，用 `eval_layers.summarize` 重算汇总，
再渲染成一张完整的归因表。

用法
────
    python3 merge_eval_reports.py /tmp/e2_a.json /tmp/e2_b.json \
        --out /tmp/e2_merged.json --layer-report /tmp/layer.md

    # 只改了用例标注（needs_inference 之类），不想重跑模型：
    python3 merge_eval_reports.py /tmp/e2_a.json /tmp/e2_b.json \
        --recases test_cases2.json --layer-report /tmp/layer.md

    ↑ 报告里已经存了每条的 answer 和 contexts，所以归因可以**离线重算**。
      改一个标注就要重跑 38 条（7 分钟）是不可接受的——这也正是
      "把证据存进报告"的第二个好处（第一个是能定位问题）。
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import eval_layers  # noqa: E402


def _load_case_index(paths):
    """把用例文件读成 id → case 的索引，供离线重新归因用。"""
    index = {}
    for p in paths:
        f = Path(p)
        if not f.is_absolute():
            f = BASE_DIR / f
        if not f.exists():
            print(f"⚠️ 用例文件不存在：{f}", file=sys.stderr)
            continue
        for c in json.loads(f.read_text(encoding="utf-8")):
            if c.get("id"):
                index[c["id"]] = c
    return index


def reattribute(details, case_index):
    """用最新的用例标注重算归因，不动 answer / score（分数只取决于答案与判据）。

    为什么不动 score：重新打分需要 must_contain 也一并更新，而那是"改判据"，
    属于另一件事（会让历史分数不可比）。这里只重算"丢在哪一层"的解释。
    """
    changed = 0
    for r in details:
        case = case_index.get(r.get("id"))
        if not case:
            continue
        fresh = eval_layers.attribute(case, r.get("answer") or "", r.get("contexts") or [])
        if fresh["layer"] != r.get("layer"):
            changed += 1
        r.update(fresh)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="合并分批评测报告并重算分层归因")
    ap.add_argument("reports", nargs="+", help="报告 JSON 路径（按顺序拼接）")
    ap.add_argument("--out", default="", help="合并后的 JSON 路径（留空则只打印）")
    ap.add_argument("--layer-report", default="", help="归因 Markdown 输出路径")
    ap.add_argument(
        "--recases",
        nargs="*",
        default=[],
        help="用例文件；给了就用最新标注重算归因（不重跑模型）",
    )
    args = ap.parse_args()

    details, files, elapsed = [], [], 0.0
    for p in args.reports:
        f = Path(p)
        if not f.exists():
            print(f"⚠️ 跳过不存在的报告：{f}", file=sys.stderr)
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        details.extend(data.get("details") or [])
        files.extend(data.get("case_files") or [])
        elapsed += float(data.get("elapsed_total") or 0.0)

    if not details:
        print("❌ 没有可合并的结果", file=sys.stderr)
        return 1

    if args.recases:
        index = _load_case_index(args.recases)
        changed = reattribute(details, index)
        print(f"🔁 按 {len(index)} 条用例的最新标注重算归因，{changed} 条标签有变化")

    # 用统一口径重算：均分、召回率、损失拆解都从 details 重算，
    # 不信任各批次自己报的平均分（分批时每批的"平均"只是那一批的）。
    summary = eval_layers.summarize(details)
    cats = {}
    for r in details:
        if r.get("layer") == "error":
            continue
        cats.setdefault(r.get("category") or "未分类", []).append(float(r.get("score") or 0.0))

    merged = {
        "case_files": sorted(set(files)),
        "total": len(details),
        "success": summary["success"],
        "failed": summary["failed"],
        "avg_score": summary["avg_score"],
        "elapsed_total": round(elapsed, 2),
        "categories": {
            c: {"count": len(s), "avg_score": round(sum(s) / len(s), 4)}
            for c, s in cats.items()
        },
        "layers": summary,
        "details": details,
    }

    print(f"📊 合并 {len(args.reports)} 份 ｜ {len(details)} 条 ｜ 均分 {summary['avg_score']:.4f}")
    recall = summary["retrieval_recall"]
    print(
        f"🧭 检索召回率 {recall:.4f}（{summary['recall_n']} 条）"
        if recall is not None
        else "🧭 检索召回率 —（无适用用例）"
    )
    print(
        f"   丢分 {summary['lost_total']:.2f} = "
        f"检索 {summary['lost_retrieval']:.2f} + 生成丢失 {summary['lost_generation']:.2f} + "
        f"多答 {summary['lost_violation']:.2f} + 未推理 {summary['lost_inference']:.2f}"
    )
    for name, info in sorted(
        summary["layers"].items(), key=lambda kv: (kv[0] not in ("ok", "n/a"), -kv[1]["lost_score"])
    ):
        print(f"   {name:<16} {info['count']:>2} 条 ｜ -{info['lost_score']:.2f} 分 ｜ {info['cases']}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✅ 已保存：{args.out}")
    if args.layer_report:
        Path(args.layer_report).write_text(
            eval_layers.render_markdown(details, summary), encoding="utf-8"
        )
        print(f"🧭 归因表已保存：{args.layer_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
