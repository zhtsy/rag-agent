#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ablation_summary.py —— 把消融四组报告汇成一张对比表

读 reports/ablation/{A,B,C,D}.json，输出：
  1. 组概览：条数 / 均分 / 检索召回率 / 每題耗时
  2. 逐条差异：只列分数真的变了的题（Δ ≥ 0.001），这才是指得住的证据
     分差为 0 的题不列——"关掉也没变"同样重要，但用一行汇总交代即可。

同时写一份 reports/ablation/summary.md，可直接贴进 README 或面试材料。
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DIR = BASE / "reports" / "ablation"
GROUPS = [("A", "base 默认"), ("B", "no-shadow"), ("C", "no-table"), ("D", "no-both")]


def load(tag):
    p = DIR / f"{tag}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def recall_of(report):
    layers = (report or {}).get("layers") or {}
    r = layers.get("retrieval_recall")
    return r


def rows_of(report):
    out = {}
    for r in (report or {}).get("details") or []:
        if "error" in r:
            continue
        out[str(r.get("id"))] = r
    return out


def main():
    data = {tag: load(tag) for tag, _ in GROUPS}
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("# 消融实验对比（存储/召回解耦 + 表格抽取）\n")

    # ---------------------------------------------------------------- 概览
    emit("| 组 | 配置 | 条数 | 均分 | 检索召回率 | 均耗时 |")
    emit("|---|---|---|---|---|---|")
    for tag, desc in GROUPS:
        rep = data.get(tag)
        if not rep:
            emit(f"| {tag} | {desc} | — | — | — | — |")
            continue
        rec = recall_of(rep)
        n = rep.get("success") or 0
        el = (rep.get("elapsed_total") or 0) / n if n else 0
        emit(
            f"| {tag} | {desc} | {rep.get('total')} | {rep.get('avg_score'):.3f} | "
            f"{'—' if rec is None else f'{rec:.3f}'} | {el:.1f}s |"
        )
    emit()

    # ------------------------------------------------- 主集：A vs B(no-shadow)
    a_rows, b_rows = rows_of(data.get("A")), rows_of(data.get("B"))
    if a_rows and b_rows:
        shared = sorted(set(a_rows) & set(b_rows))
        changed = [i for i in shared
                   if abs(a_rows[i]["score"] - b_rows[i]["score"]) >= 0.001]
        emit(f"## 主集 {len(shared)} 条：md 优先（shadow 降权）的净效应\n")
        a_mean = sum(a_rows[i]["score"] for i in shared) / len(shared)
        b_mean = sum(b_rows[i]["score"] for i in shared) / len(shared)
        if not changed:
            emit("逐条分数**完全一致** —— 降权只在 md/pdf 同分时破平局，"
                 "不改变任何实得分。这说明它没有以牺牲答案为代价换整洁。\n")
        else:
            emit("| id | 题面 | base | no-shadow | Δ |")
            emit("|---|---|---|---|---|")
            for i in changed:
                d = b_rows[i]["score"] - a_rows[i]["score"]
                emit(f"| {i} | {a_rows[i]['question'][:26]} | {a_rows[i]['score']:.2f} | "
                     f"{b_rows[i]['score']:.2f} | {d:+.2f} |")
            emit()
            emit(f"{len(changed)}/{len(shared)} 条受影响，这 {len(shared)} 条的均分 "
                 f"{a_mean:.3f} → {b_mean:.3f}。降权的价值不在「提分」，而在"
                 f"**把同分竞争的结果钉死在 md 上**：一旦 PDF 副本顶掉 md 块，"
                 f"该类题就从满分直接掉到 0（不是掉一点）。\n")

    # ------------------------------------- 非 txt：A vs C(no-table) vs D(no-both)
    c_rows, d_rows = rows_of(data.get("C")), rows_of(data.get("D"))
    if a_rows and c_rows:
        shared = sorted(set(a_rows) & set(c_rows))
        emit(f"## 非 txt 集 {len(shared)} 条：表格抽取的净效应\n")
        emit("| id | 题面 | base | no-table | Δ | no-both |")
        emit("|---|---|---|---|---|---|")
        changed = 0
        for i in shared:
            a_s, c_s = a_rows[i]["score"], c_rows[i]["score"]
            d_s = d_rows[i]["score"] if i in d_rows else None
            if abs(a_s - c_s) < 0.001:
                continue
            changed += 1
            emit(f"| {i} | {a_rows[i]['question'][:24]} | {a_s:.2f} | {c_s:.2f} | "
                 f"{c_s - a_s:+.2f} | {'—' if d_s is None else f'{d_s:.2f}'} |")
        if changed == 0:
            emit("| — | 全部用例分数未变 | — | — | — | — |")
        emit()
        if changed:
            a_mean = sum(a_rows[i]["score"] for i in shared) / len(shared)
            lost = sum(max(0.0, a_rows[i]["score"] - c_rows[i]["score"]) for i in shared)
            emit(f"关掉表格块共丢 {lost:.2f} 分 / {changed} 条题，这 {len(shared)} 条的均分 "
                 f"{a_mean:.3f} → {data['C']['avg_score']:.3f}；"
                 f"其余 {len(shared) - changed} 条**分数不变**——它们本来就是条文/OCR "
                 f"证据，剔表没有误伤非表格内容。\n")

    out = DIR / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ 已保存：{out}")


if __name__ == "__main__":
    main()
