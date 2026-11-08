# -*- coding: utf-8 -*-
"""eval_layers.py —— 把「丢的分」归因到检索层还是生成层。

要解决的问题
────────────
一次评测只报「T028 得 0 分」没有可操作信息：既可能是**没检索到证据**，
也可能是**证据在、模型没写进去**，还可能是**写多了**。三种情况的修法完全不同：
改召回 / 改选块与提示词 / 加约束。而归因所需的材料本来就在手上——
`/chat/stream` 的收尾帧一直下发着 `contexts`（本轮检索到的原始文档块），
只是评测脚本以前**把它扔了**。

本模块只做纯字符串比对，不联网、不加载模型、不依赖第三方库，
所以能被 pytest 毫秒级覆盖，也能在 CI 里跑。

归因口径
────────
对每条用例的每个 `must_contain` 关键词，判断它在不在**答案**里、在不在**证据**里：

┌────────────────┬──────────┬──────────┬──────────────────────────┐
│ 在答案里       │ 在证据里 │ 归因     │ 该改哪里                 │
├────────────────┼──────────┼──────────┼──────────────────────────┤
│ 是             │ 是       │ ok       │ —                        │
│ 是             │ 否       │ ok       │ 模型自己知道，非检索贡献 │
│ 否             │ 是       │ generation_loss │ 提示词 / 选块 / 截断  │
│ 否             │ 否       │ retrieval_miss  │ 召回 / 分块 / 图谱补边 │
└────────────────┴──────────┴──────────┴──────────────────────────┘

两类特例
────────
1. **拒答类**（category == "拒答"）：`must_contain` 是"未找到/无法"这类拒答措辞，
   本来就不该出现在证据里，做召回率没有意义 → 归因记 `n/a`，
   但 `must_not_contain` 命中（真把密码编出来了）仍然记 `violation`。

2. **需要一步推理的用例**（用例里标 `"needs_inference": true`）：
   `must_contain` 是推出来的结论（"南京"→"江苏"、"7 点加 1 小时"→"8"），
   按字面永远搜不到。这种不该算检索的锅 → 单独记 `inference_miss`，
   语义是"证据够了，但模型没走那一步（或按策略拒答）"。

   ⚠️ 这个标记必须由人给出，不能靠猜。它是**用例的元数据**，不是脚本的推断：
   只有出题人知道某个词是不是必须从原文推出来。所以别在代码里塞启发式规则。

检索召回率（`recall`）
──────────────────────
`recall = 命中的 must_contain 数 / len(must_contain)`，只看证据、不看生成。
它是本模块最有价值的产出：**不随模型输出抖动**。端到端分数会因为模型换了个
说法在 0.5~1.0 之间跳，召回率不会——它才适合放进回归门禁。

用法
────
    from eval_layers import attribute, summarize, render_markdown
    row = attribute(case, answer, contexts)
    row["layer"]          # 'ok' / 'retrieval_miss' / 'generation_loss' / ...
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

# 拒答判定复用 confidence.py 的唯一实现（纯函数、只 import re），不在这里再写一份。
# 为什么归因需要它：拒答时模型会把问题里的实体原样复述一遍——
# "知识库中没有马斯克的信息" 里就含 "马斯克"。按字面 `kw in answer` 会算成"答到了"，
# 于是归因判成 ok、召回率虚高 1.0，把一个**漏召回**的问题伪装成好结果。
from confidence import looks_like_abstention as _looks_like_abstention

# 归因标签 → 人话。顺序即报告里的展示顺序（越靠前越"正常"）
LAYERS = {
    "ok": "答对",
    "violation": "生成多答（命中 must_not_contain）",
    "inference_miss": "证据在但未推出结论",
    "generation_loss": "召回但生成丢",
    "retrieval_miss": "检索未召回",
    "mixed": "混合（既漏召回也丢生成）",
    "n/a": "不适用（拒答类）",
    "error": "请求失败",
}

# 会「丢分」的归因，用于排序与汇总
DEFECT_LAYERS = ("mixed", "retrieval_miss", "generation_loss", "inference_miss", "violation")

ABSTAIN_CATEGORY = "拒答"


# --------------------------------------------------------------- 文本预处理
def _norm(text: str) -> str:
    """统一小写比较。

    模型对同一个词的大小写并不稳定（同一题实测既写 "Python" 也写 "python"），
    而用例里写的是小写——用区分大小写的 `in` 会把完全正确的答案判成漏答。
    （T001「小智喜欢什么」历史上就这么被误判成 0.50。）
    """
    return (text or "").lower()


def strip_header(context: str) -> str:
    """去掉 `【来源：xxx】【置信度：0.9】` 这类头部，只留正文。

    为什么必须去掉：来源文件名本身可能撞上关键词（问"test1 里写了什么"、
    或用例恰好要匹配文件名），而那是**元信息**不是**证据正文**。
    更实际的原因：头部里有数字（置信度 0.9966），做数字类关键词比对时会误命中。
    """
    body = (context or "").lstrip()
    if body.startswith("【来源："):
        # 头部可能跨多行（图谱块是 "【来源：知识图谱属性】【确定性事实】\n正文"），
        # 但一定以第一个换行为界——头部本身不含换行。
        _, _, rest = body.partition("\n")
        return rest.strip()
    return body.strip()


def context_body(contexts: Optional[Sequence[str]]) -> str:
    """把若干证据块拼成一段可检索的文本（已去头部、已小写）。"""
    if not contexts:
        return ""
    return _norm("\n".join(strip_header(c) for c in contexts if c))


# ------------------------------------------------------------------- 单条归因
def attribute(
    case: Dict[str, Any],
    answer: str,
    contexts: Optional[Sequence[str]],
    *,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """给一条评测结果打归因标签。

    返回的字典直接并进评测明细，字段全部可 JSON 序列化：
        layer            归因标签（见 LAYERS）
        recall           检索召回率（拒答类为 None）
        in_answer        答出来的关键词
        missing_answer   没答出来的关键词
        lost_in_retrieval 没答出来、且证据里也没有的关键词
        lost_in_generation 证据里有、却没答出来的关键词
        unsupported_hits  答出来了、但证据里没有的关键词（模型自带的记忆）
        evidence_n       证据块数
    """
    must: List[str] = list(case.get("must_contain") or [])
    forbid: List[str] = list(case.get("must_not_contain") or [])
    is_abstain = case.get("category") == ABSTAIN_CATEGORY

    if error:
        return {
            "layer": "error",
            "recall": None,
            "in_answer": [],
            "missing_answer": must,
            "lost_in_retrieval": [],
            "lost_in_generation": [],
            "unsupported_hits": [],
            "evidence_n": len(contexts or []),
        }

    ans = _norm(answer)
    ctx = context_body(contexts)

    in_context = [kw for kw in must if _norm(kw) in ctx]

    # 拒答时不允许"复述实体"冒充命中（见文件顶部 import 处的说明）。
    # 判据用的是线上同一份拒答词表，不用另造一套。
    abstained_answer = bool(answer) and _looks_like_abstention(answer)
    if abstained_answer and not is_abstain:
        in_answer: List[str] = []
    else:
        in_answer = [kw for kw in must if _norm(kw) in ans]

    missing_answer = [kw for kw in must if _norm(kw) not in ans]
    if abstained_answer and not is_abstain:
        missing_answer = list(must)
    # 是否出现在证据里，只对「没答出来」的词有归因意义：
    # 答出来的词即使证据里没有（模型自带记忆），也不构成"漏召回"的缺陷。
    lost_in_retrieval = [kw for kw in missing_answer if _norm(kw) not in ctx]
    lost_in_generation = [kw for kw in missing_answer if _norm(kw) in ctx]
    unsupported_hits = [kw for kw in in_answer if _norm(kw) not in ctx]

    violation = [kw for kw in forbid if _norm(kw) in ans]

    # ---- 标签 ----
    if violation:
        # 多答是最硬的缺陷：即使要点都答到了，也不该给"答对"。
        # 拒答类尤其严重——真把银行卡密码编出来了。
        layer = "violation"
    elif is_abstain:
        layer = "n/a"
    elif not missing_answer:
        layer = "ok"
    elif lost_in_retrieval and lost_in_generation:
        layer = "mixed"
    elif lost_in_retrieval:
        layer = "inference_miss" if case.get("needs_inference") else "retrieval_miss"
    else:
        layer = "generation_loss"

    # ---- 召回率：只看证据，不看生成 ----
    # 分母是 must_contain 的词数，分子是「证据里真的出现过」的词数。
    # 注意"答出来了但证据里没有"（unsupported_hits）不算召回命中——
    # 那是模型自带的记忆，不是检索的功劳，算进去会让召回率虚高。
    if is_abstain or not must:
        recall = None
    else:
        recall = round(len(in_context) / len(must), 4)

    return {
        "layer": layer,
        "recall": recall,
        "in_answer": in_answer,
        "in_context": in_context,
        "missing_answer": missing_answer,
        "lost_in_retrieval": lost_in_retrieval,
        "lost_in_generation": lost_in_generation,
        "unsupported_hits": unsupported_hits,
        "violation_words": violation,
        "evidence_n": len(contexts or []),
    }


# ------------------------------------------------------------------- 汇总
def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """把一批归因结果汇成报告用的统计。

    rows 里每项需要有：layer / score / recall（可缺）。
    """
    rows = list(rows)
    ok_rows = [r for r in rows if r.get("layer") != "error"]

    by_layer: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok_rows:
        by_layer.setdefault(r.get("layer") or "n/a", []).append(r)

    layers = {}
    for name, group in by_layer.items():
        lost = sum(1 - float(r.get("score") or 0.0) for r in group)
        layers[name] = {
            "count": len(group),
            "label": LAYERS.get(name, name),
            "lost_score": round(lost, 4),
            "cases": [r.get("id") for r in group],
        }

    recalls = [r["recall"] for r in ok_rows if r.get("recall") is not None]
    scores = [float(r.get("score") or 0.0) for r in ok_rows]

    n = len(ok_rows)
    return {
        "total": len(rows),
        "success": n,
        "failed": len(rows) - n,
        "avg_score": round(sum(scores) / n, 4) if n else 0.0,
        "retrieval_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "recall_n": len(recalls),
        "layers": layers,
        # 三类缺陷各自「吃掉」了多少分——这是这张表存在的理由
        "lost_total": round(sum(1 - s for s in scores), 4),
        "lost_retrieval": round(
            sum(
                1 - float(r.get("score") or 0.0)
                for r in ok_rows
                if r.get("layer") in ("retrieval_miss", "mixed")
            ),
            4,
        ),
        "lost_generation": round(
            sum(
                1 - float(r.get("score") or 0.0)
                for r in ok_rows
                if r.get("layer") == "generation_loss"
            ),
            4,
        ),
        "lost_violation": round(
            sum(
                1 - float(r.get("score") or 0.0)
                for r in ok_rows
                if r.get("layer") == "violation"
            ),
            4,
        ),
        # 推理题单独列：它的修法既不是改召回也不是改提示词，而是产品决策
        # （"要不要允许模型跨出知识库做一步推理"）。
        "lost_inference": round(
            sum(
                1 - float(r.get("score") or 0.0)
                for r in ok_rows
                if r.get("layer") == "inference_miss"
            ),
            4,
        ),
    }


def _hint(row: Dict[str, Any]) -> str:
    """给缺陷行配一句"该改什么"，比让人对着几个标签字段自己推要好。"""
    layer = row.get("layer")
    if layer == "violation":
        return "答了不该答的：" + "、".join(row.get("violation_words") or [])
    if layer in ("retrieval_miss", "mixed"):
        words = row.get("lost_in_retrieval") or []
        return "证据里也没有：" + "、".join(words) if words else "证据里也没有"
    if layer == "generation_loss":
        words = row.get("lost_in_generation") or []
        return "证据在、答案没写：" + "、".join(words) if words else "证据在、答案没写"
    if layer == "inference_miss":
        words = row.get("lost_in_retrieval") or []
        # 这里必须说清"不是召回的锅"，否则看表的人会去优化召回——永远修不好。
        return "需一步外部推理（证据只有中间事实），缺：" + "、".join(words)
    return ""


def render_markdown(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    """生成可直接贴进 PR / README 的归因表。"""
    lines: List[str] = []
    add = lines.append

    add("# 评测分层归因报告")
    add("")
    add(
        "由 `eval_system.py --layer-report` 或 `merge_eval_reports.py` 生成。"
        "归因口径见 `eval_layers.py` 顶部说明。"
    )
    add("")

    add("## 总览")
    add("")
    add("| 指标 | 值 |")
    add("| :--- | ---: |")
    add(f"| 用例数 | {summary['total']}（成功 {summary['success']}） |")
    add(f"| 端到端均分 | **{summary['avg_score']:.4f}** |")
    recall = summary.get("retrieval_recall")
    add(
        f"| 检索召回率 | **{recall:.4f}**（{summary['recall_n']} 条非拒答） |"
        if recall is not None
        else "| 检索召回率 | —（无适用用例） |"
    )
    add(f"| 丢掉的总分 | {summary['lost_total']:.4f} |")
    add(f"| ├ 检索未召回 | {summary['lost_retrieval']:.4f} |")
    add(f"| ├ 生成丢失 | {summary['lost_generation']:.4f} |")
    add(f"| ├ 生成多答 | {summary['lost_violation']:.4f} |")
    add(f"| └ 未走推理 | {summary['lost_inference']:.4f} |")
    add("")

    add("## 按归因分布")
    add("")
    add("| 归因 | 条数 | 吃掉的分 | 用例 |")
    add("| :--- | ---: | ---: | :--- |")
    ordered = sorted(
        summary["layers"].items(),
        key=lambda kv: (kv[0] not in ("ok", "n/a"), -kv[1]["lost_score"]),
    )
    for name, info in ordered:
        cases = ", ".join(str(c) for c in info["cases"]) or "—"
        add(f"| `{name}` {info['label']} | {info['count']} | {info['lost_score']:.2f} | {cases} |")
    add("")

    defect = [r for r in rows if r.get("layer") in DEFECT_LAYERS]
    if defect:
        add("## 缺陷明细")
        add("")
        add("| 用例 | 归因 | 得分 | 召回 | 该改什么 |")
        add("| :--- | :--- | ---: | ---: | :--- |")
        for r in sorted(defect, key=lambda x: float(x.get("score") or 0.0)):
            rec = r.get("recall")
            add(
                "| {id} | `{layer}` | {score:.2f} | {rec} | {hint} |".format(
                    id=r.get("id"),
                    layer=r.get("layer"),
                    score=float(r.get("score") or 0.0),
                    rec=f"{rec:.2f}" if rec is not None else "—",
                    hint=_hint(r) or "—",
                )
            )
        add("")

    return "\n".join(lines) + "\n"
