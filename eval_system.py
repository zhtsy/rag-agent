#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_system.py —— 离线批量评测（HTTP 版）

和原来本地那版的区别（重要）
────────────────────────────
原版是 `from rag_agent import run_conversation`，在评测进程里**再加载一遍**
bge + reranker（约 1.5G），而本机只有 2 vCPU / 3G 内存，历史上就是这么把
backend 挤爆的（OOM）。

这一版改成走 HTTP 打 `/chat/stream`：
  · 零额外内存，不会跟 backend 抢模型；
  · 评的是线上真实链路（含流式、护栏、事实核查），不是进程内副本；
  · 只用标准库，宿主机/容器里都能直接跑。

评分口径与原版一致
──────────────────
    hit_rate       = 命中 must_contain 的比例（拒答类只要命中任意一个即 1.0）
    violation_rate = 命中 must_not_contain 的比例
    score          = hit_rate * (1 - violation_rate)
    报告           = 总分 + 分类均分 + 明细 + 失败清单

分层归因（--layer-report）
──────────────────────────
总分只说"丢了多少"，说不出"丢在哪一层"。所以每条用例现在会把本轮
`contexts`（检索到的原始文档块）一起存进报告，并用 `eval_layers.attribute`
判一次归因：

    retrieval_miss   证据里根本没有 → 改召回 / 分块 / 图谱补边
    generation_loss  证据里有但没写进答案 → 改提示词 / 选块顺序 / 上下文截断
    violation        答了不该答的 → 加约束

顺带产出**检索召回率**：只看证据、不看生成，不随模型换说法而抖动，
比端到端分数更适合做回归门禁。

用法
────
    python3 eval_system.py                          # 跑默认两个测试集
    python3 eval_system.py --cases test_cases2.json
    python3 eval_system.py --limit 5 --sleep 1      # 先抽 5 条试水
    python3 eval_system.py --base http://127.0.0.1:8000
    python3 eval_system.py --cases test_cases2.json --layer-report layer.md
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = ["test_cases2.json", "test_cases3.json"]

# 后端 /chat/stream 已加鉴权；密钥从 api_auth 统一读取（环境变量或 .rag_api_key）
sys.path.insert(0, str(BASE_DIR))
try:
    from api_auth import auth_headers as _auth_headers
except Exception:  # noqa: BLE001
    def _auth_headers() -> dict:
        return {}

# 分层归因：纯字符串比对，零第三方依赖（见 eval_layers.py 顶部口径说明）
import eval_layers


# ---------------------------------------------------------------- HTTP / SSE
def ask_stream(base: str, question: str, session_id: str, timeout: int = 180,
               max_retries: int = 3):
    """打 /chat/stream，返回 (answer, sources, confidence, contexts)。

    429 是**并发闸门的正常回执**（后端同时在跑的请求到上限了，见 main1.py）。
    评测是顺序跑的，但只要有人同时在用前端就可能撞上。这里按 `Retry-After` 重试几次，
    而不是把它记成一条 0 分的"失败行"——那会凭空造出一个不存在的缺陷，
    而且下次重跑就消失，最难查的那种。
    """
    url = base.rstrip("/") + "/chat/stream"
    body = json.dumps({"message": question, "session_id": session_id}).encode("utf-8")

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json", **_auth_headers()}
        )
        try:
            chunks, sources, contexts, confidence = [], [], [], None
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
                    if "sources" in obj:
                        sources = obj["sources"] or []
                    if "contexts" in obj:
                        contexts = obj["contexts"] or []
                    if "confidence" in obj:
                        confidence = obj["confidence"]
            return "".join(chunks), sources, confidence, contexts
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt >= max_retries:
                raise
            wait = float(e.headers.get("Retry-After") or 5)
            print(f"    ⏳ 429 过载，{wait:.0f}s 后重试（第 {attempt + 1}/{max_retries} 次）")
            time.sleep(wait)

    raise RuntimeError("unreachable")


# ------------------------------------------------------------------- scoring
def score_case(case: dict, answer: str):
    must = case.get("must_contain") or []
    forbid = case.get("must_not_contain") or []

    # 关键词匹配统一按小写比较。
    # 模型对同一个词的大小写并不稳定（同一题实测会写 "Python" 也会写 "python"），
    # 而用例里写的是小写，用 `kw in answer` 区分大小写会把完全正确的答案判成漏答
    # —— T001「小智喜欢什么」就这么被误判成 0.50。
    al = answer.lower()

    if case.get("category") == "拒答":
        # 拒答类不要求覆盖全部关键词，命中任意一个说明"该拒的拒对了"
        hit = 1 if any(kw.lower() in al for kw in must) else 0
        hit_rate = float(hit)
    else:
        hit = sum(1 for kw in must if kw.lower() in al)
        hit_rate = hit / len(must) if must else 1.0

    violation = sum(1 for kw in forbid if kw.lower() in al)
    violation_rate = violation / len(forbid) if forbid else 0.0

    return hit_rate * (1 - violation_rate), hit_rate, violation_rate, hit, violation


def load_cases(paths):
    cases, loaded = [], []
    for p in paths:
        f = Path(p)
        if not f.is_absolute():
            f = BASE_DIR / f
        if not f.exists():
            print(f"⚠️ 跳过不存在的测试集：{f}", file=sys.stderr)
            continue
        with f.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for c in data:
            c["_file"] = f.name
            cases.append(c)
        loaded.append(f.name)
    return cases, loaded


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG 离线批量评测（HTTP 版）")
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="后端地址")
    ap.add_argument("--cases", nargs="+", default=DEFAULT_CASES, help="测试集 JSON，可多个")
    ap.add_argument("--out", default="eval_report.json", help="报告输出路径")
    ap.add_argument("--limit", type=int, default=0, help="最多跑多少条（0=全部）")
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 条（长测试集分批跑用）")
    ap.add_argument("--sleep", type=float, default=2.0, help="每问之间的间隔秒数")
    ap.add_argument("--timeout", type=int, default=180, help="单问超时秒数")
    ap.add_argument(
        "--layer-report",
        default="",
        help="额外输出分层归因报告（Markdown 路径；留空则只在终端打印汇总）",
    )
    args = ap.parse_args()

    cases, files = load_cases(args.cases)
    if not cases:
        print("❌ 没有可用的测试集", file=sys.stderr)
        return 1
    if args.offset > 0:
        cases = cases[args.offset:]
    if args.limit > 0:
        cases = cases[: args.limit]

    print(f"📋 测试集：{', '.join(files)} ｜ 本次 {len(cases)} 条 ｜ 后端 {args.base}")
    print("=" * 60)

    results, total_score, t_all = [], 0.0, time.time()
    for i, case in enumerate(cases, 1):
        qid = case.get("id", f"#{i}")
        q = case["question"]
        print(f"[{i}/{len(cases)}] {qid} {q}")
        t0 = time.time()
        try:
            answer, sources, confidence, ctx = ask_stream(
                args.base, q, f"eval-{qid}", timeout=args.timeout
            )
        except urllib.error.URLError as e:
            print(f"    ❌ 请求失败：{e}")
            results.append({
                "id": qid, "category": case.get("category"), "score": 0.0,
                "error": str(e), "elapsed": round(time.time() - t0, 2),
                **eval_layers.attribute(case, "", [], error=str(e)),
            })
            time.sleep(args.sleep)
            continue

        elapsed = time.time() - t0
        score, hit_rate, vio_rate, hit, viol = score_case(case, answer)
        total_score += score

        # ★ 分层归因：拿本轮检索到的原始文档块跟答案比对，
        #   判断丢的分是「检索没召回」还是「召回但生成丢了」。
        layer = eval_layers.attribute(case, answer, ctx)

        print(f"    👍 {hit}/{len(case.get('must_contain') or [])} 命中 ｜ "
              f"🚫 {viol} 违规 ｜ conf={confidence} ｜ {elapsed:.1f}s ｜ 得分 {score:.2f}"
              f" ｜ 归因 {layer['layer']}")
        print(f"    A: {answer[:70]}")

        results.append({
            "id": qid,
            "category": case.get("category"),
            "file": case.get("_file"),
            "question": q,
            "answer": answer,
            "answer_len": len(answer),
            "sources": sources,
            "confidence": confidence,
            "hit": hit,
            "hit_rate": round(hit_rate, 4),
            "violation": viol,
            "violation_rate": round(vio_rate, 4),
            "score": round(score, 4),
            "elapsed": round(elapsed, 2),
            # ★ 证据链：以前这里只有文件名，归因只能靠人读答案猜。
            #   现在把原始文档块一起存进报告——它是"检索丢了还是生成丢了"的唯一判据。
            "contexts": ctx,
            **layer,
        })
        time.sleep(args.sleep)

    # ------------------------------------------------------------- 汇总
    ok = [r for r in results if "error" not in r]
    avg = total_score / len(cases) if cases else 0.0

    cats = {}
    for r in ok:
        cats.setdefault(r.get("category") or "未分类", []).append(r["score"])

    print("=" * 60)
    print(f"📊 总条数 {len(cases)} ｜ 成功 {len(ok)} ｜ 失败 {len(cases) - len(ok)}")
    print(f"📊 平均得分 {avg:.3f} ｜ 总耗时 {time.time() - t_all:.1f}s")
    for cat, scores in sorted(cats.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"    {cat:<12} 均分 {sum(scores) / len(scores):.3f}  ({len(scores)} 条)")

    weak = sorted(ok, key=lambda r: r["score"])[:5]
    if weak and weak[0]["score"] < 0.8:
        print("\n🔎 最低分 5 条（优先排查）：")
        for r in weak:
            print(f"    {r['score']:.2f}  {r['id']:<6} {r['question'][:40]}")
            print(f"           A: {r['answer'][:60]}")

    # ------------------------------------------------------- 分层归因汇总
    layer_sum = eval_layers.summarize(results)
    print("\n" + "=" * 60)
    print("🧭 分层归因（丢的分到底丢在哪一层）")
    recall = layer_sum["retrieval_recall"]
    print(f"    检索召回率 {recall:.3f}（{layer_sum['recall_n']} 条非拒答，只看证据不看生成）"
          if recall is not None else "    检索召回率 —（无适用用例）")
    print(f"    总共丢掉 {layer_sum['lost_total']:.2f} 分："
          f"检索未召回 {layer_sum['lost_retrieval']:.2f} ｜ "
          f"生成丢失 {layer_sum['lost_generation']:.2f} ｜ "
          f"生成多答 {layer_sum['lost_violation']:.2f}")
    for name, info in sorted(layer_sum["layers"].items(),
                             key=lambda kv: (kv[0] not in ("ok", "n/a"), -kv[1]["lost_score"])):
        if name in ("ok", "n/a"):
            continue
        print(f"    ⚠️ {name:<16} {info['count']:>2} 条 ｜ -{info['lost_score']:.2f} 分 ｜ "
              f"{', '.join(str(c) for c in info['cases'])}")

    # 缺陷明细：只列真正丢分的，避免刷屏
    defect = [r for r in ok if r.get("layer") in eval_layers.DEFECT_LAYERS]
    if defect:
        print("\n🔬 缺陷明细（证据在不在，决定了该改哪一层）：")
        for r in sorted(defect, key=lambda x: x["score"]):
            print(f"    {r['score']:.2f}  {r['id']:<6} {r['layer']:<15} "
                  f"召回 {r['recall'] if r['recall'] is None else format(r['recall'], '.2f')}")
            if r.get("lost_in_retrieval"):
                print(f"            证据里也没有：{'、'.join(r['lost_in_retrieval'])}")
            if r.get("lost_in_generation"):
                print(f"            证据在却丢：  {'、'.join(r['lost_in_generation'])}")

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base": args.base,
        "case_files": files,
        "total": len(cases),
        "success": len(ok),
        "avg_score": round(avg, 4),
        "elapsed_total": round(time.time() - t_all, 2),
        "categories": {
            c: {"count": len(s), "avg_score": round(sum(s) / len(s), 4)}
            for c, s in cats.items()
        },
        # ★ 分层归因汇总：端到端分数之外，单独给出「检索召回率」与损失拆解
        "layers": layer_sum,
        "details": results,
    }

    out = Path(args.out)
    if not out.is_absolute():
        out = BASE_DIR / out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 报告已保存：{out}")

    if args.layer_report:
        lr = Path(args.layer_report)
        if not lr.is_absolute():
            lr = BASE_DIR / lr
        lr.write_text(eval_layers.render_markdown(results, layer_sum), encoding="utf-8")
        print(f"🧭 归因报告已保存：{lr}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
