# -*- coding: utf-8 -*-
"""
flywheel_analyzer.py —— 数据飞轮 · 带证据的 Bad Case 归因（v2）

和 v1 的本质区别
────────────────
v1 只把「问题 + 回答」丢给模型，模型看不到检索原文，只能靠语气猜，
结果几乎全被判成「检索未命中」——归因等于没归因。

v2 把【检索到的上下文】一起送进去，让模型做**事实比对**：
    · 回答能在上下文里逐句找到依据？        → 无效反馈/用户主观不满（系统没毛病，别改）
    · 上下文有料，但回答无中生有/自相矛盾？  → 生成幻觉
    · 上下文里压根没有相关信息？            → 检索未命中
    · 回答是乱码/结构损坏/标签泄漏？        → 格式错误

这样产出的报告才是"能派活"的：无效反馈直接忽略，有效 bad case 才进优化队列。

用法
────
    python flywheel_analyzer.py                        # 全量分析
    python flywheel_analyzer.py --limit 50 --workers 4
    python flywheel_analyzer.py --include-no-context    # 连老数据（无 contexts）也强行判
    python flywheel_analyzer.py --log feedback_log.jsonl --out flywheel_report.md

依赖
────
    pip install openai python-dotenv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# =====================================================================
# 0. 常量
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_LOG = BASE_DIR / "feedback_log.jsonl"
DEFAULT_OUT = BASE_DIR / "flywheel_report.md"

# ---- 四个归因标签（顺序＝报告里的展示顺序）----
LABEL_RETRIEVAL_MISS = "检索未命中"
LABEL_HALLUCINATION = "生成幻觉"
LABEL_FORMAT = "格式错误"
LABEL_INVALID = "无效反馈/用户主观不满"

LABELS = [LABEL_RETRIEVAL_MISS, LABEL_HALLUCINATION, LABEL_FORMAT, LABEL_INVALID]
VALID_LABELS = [LABEL_RETRIEVAL_MISS, LABEL_HALLUCINATION, LABEL_FORMAT]   # 真·Bad Case

LABEL_ICON = {
    LABEL_RETRIEVAL_MISS: "🔍",
    LABEL_HALLUCINATION: "🌀",
    LABEL_FORMAT: "🧩",
    LABEL_INVALID: "🙈",
}

# 标签归一化关键词表：模型不听话、输出自由文本时，用它把结果拉回四分类
LABEL_KEYWORDS = {
    LABEL_INVALID: ["无效反馈", "主观", "误点", "误判", "用户不满", "忽略该反馈", "无需优化",
                    "忠于上下文", "有依据", "回答正确", "invalid", "subjective", "no issue"],
    LABEL_RETRIEVAL_MISS: ["检索未命中", "未命中", "召回", "没检索到", "上下文为空", "无关",
                           "不相关", "知识库没有", "retrieval miss", "no context"],
    LABEL_HALLUCINATION: ["幻觉", "编造", "杜撰", "无中生有", "与上下文矛盾", "上下文没有提到",
                          "不实", "hallucin", "fabricat"],
    LABEL_FORMAT: ["格式错误", "乱码", "结构损坏", "标签泄漏", "截断", "不完整", "解析失败",
                   "无法阅读", "format", "malformed"],
}

# 送进 Prompt 的上下文体积上限（控制 token；超出会注明"已截断"）
MAX_CTX_PER_ITEM = 1600
MAX_CTX_TOTAL = 12000
# 报告里每条的答案/上下文摘录长度
ANSWER_EXCERPT_CHARS = 600
CTX_EXCERPT_CHARS = 900


# =====================================================================
# 1. Prompt：这是整个飞轮的大脑
# =====================================================================
SYSTEM_PROMPT = """你是一名 RAG 系统的质量审计员（QA Auditor）。你的唯一职责是：
基于「用户问题」「检索到的上下文」「AI 回答」做**事实比对**，判断这条"用户点了踩"的记录
到底是不是系统的问题。

核心原则（非常重要）：
1. 用户点了踩 ≠ 系统有错。用户可能是误点、情绪化、期望过高，或者答案本身是对的但没戳中他。
2. 你没有看到的东西，不要假设它存在。一切以【检索到的上下文】为准。
3. 你的判定必须可复核：要么引用上下文原句，要么明确指出上下文里缺什么。
4. 禁止为了"显得有用"而硬编优化建议。如果系统没错，就直说"忽略该反馈"。

【判定顺序】严格按顺序执行，第一个命中的就是结论：

第 1 步 · 检索是否命中？
   若【检索到的上下文】为空、或内容与问题毫无关系（只是字面撞词、讲的是别的话题）、
   或上下文里明确写着"未在知识库中找到相关信息" →
   → 标签 = 检索未命中。（此时回答写得再漂亮都没有依据，根因在检索/索引层）

第 2 步 · 回答是否可用？
   上下文有料，但回答是不可读/不可用的形态：乱码、JSON 或表格结构损坏、
   标签泄漏（出现 answer: / sources: / contexts: 之类的字段名）、被截断在半句话上、
   只有工具调用没有正文 →
   → 标签 = 格式错误。
   （注意：答案"简短"、"排版朴素"、"语气生硬"都不算格式错误。）

第 3 步 · 回答是否忠于上下文？
   上下文有料、回答也读得懂，但回答里的关键事实（数字、人名、时间、结论、因果关系）
   在上下文中找不到依据，或与上下文明显矛盾（比如上下文说"3 个"，回答写"5 个"）→
   → 标签 = 生成幻觉。必须指出具体是哪一句、上下文里实际写的是什么。

第 4 步 · 以上都不是 → 系统没错。
   回答的每一个关键事实都能在上下文中找到出处，并且确实正面回答了问题。
   用户点踩的原因只可能是主观不满（答案不合预期、嫌啰嗦、想要别的角度等）→
   → 标签 = 无效反馈/用户主观不满。
   此时你的建议必须是"忽略该反馈"，并说明你核对到的事实依据。

只输出一个合法 JSON 对象，不要 ``` 代码块，不要任何解释性文字：
{
  "label": "检索未命中 | 生成幻觉 | 格式错误 | 无效反馈/用户主观不满 （四选一，原样输出）",
  "verdict": "事实判定，不超过 120 字。必须引用上下文原句（用「」括起来）或明确指出上下文缺少什么。",
  "evidence": "上下文里最能支撑（或最能戳穿）这个回答的一句原话，不超过 80 字；确实没有就写 无",
  "suggestion": "若为有效 Bad Case：给出具体可执行的优化动作（改哪个环节、怎么改）。若为无效反馈：写「忽略该反馈」并说明理由。不超过 100 字。"
}"""

USER_TEMPLATE = """请审计下面这条被用户点踩的 RAG 问答，以 JSON 格式返回结论。

【用户问题】
{question}

【检索到的上下文】（共 {ctx_count} 块，这是模型当时能看到的全部依据）
{contexts}

【AI 回答】
{answer}

【用户备注】
{comment}

请严格按「判定顺序」执行：先看检索是否命中，再看格式是否可用，再做事实比对，
最后才考虑"用户主观不满"。记住：没有依据的判断一律不许下。"""


# =====================================================================
# 2. 数据读取 / 去重
# =====================================================================
def load_feedback(path: Path) -> Tuple[List[dict], int, int, int]:
    """读取 JSONL，返回 (全部记录, good 数, bad 数, 坏行数)。"""
    if not path.exists():
        print(f"❌ 找不到日志文件：{path}", file=sys.stderr)
        print("   先在页面上点几次 👎，或检查 --log 参数。", file=sys.stderr)
        sys.exit(1)

    records: List[dict] = []
    good = bad = broken = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                broken += 1
                continue
            if not isinstance(obj, dict):
                broken += 1
                continue
            records.append(obj)
            if obj.get("feedback") == "good":
                good += 1
            elif obj.get("feedback") == "bad":
                bad += 1

    if broken:
        print(f"⚠️ 跳过 {broken} 行非法 JSON（可能是写日志时进程被杀）", file=sys.stderr)
    return records, good, bad, broken


def _contexts_digest(contexts: List[str]) -> str:
    """给上下文取个短指纹，用于去重 key —— 同样的问答配不同上下文，应该分开审计。"""
    joined = "\n".join(contexts)[:600]
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:8]


def dedupe_bad_cases(bad_cases: List[dict]) -> List[dict]:
    """合并完全相同的 (问题, 回答, 上下文)，保留出现次数。

    为什么要合并：同一个人反复点踩、或同一个 bug 被多个 session 触发，
    本质是同一条 case，重复送模型纯烧 token。
    """
    buckets: "OrderedDict[Tuple[str, str, str], dict]" = OrderedDict()

    for case in bad_cases:
        question = (case.get("question") or "").strip()
        answer = (case.get("answer") or "").strip()
        contexts = case.get("contexts") or []
        if not isinstance(contexts, list):
            contexts = [str(contexts)]
        contexts = [str(c) for c in contexts if str(c).strip()]

        key = (question, answer, _contexts_digest(contexts))

        if key not in buckets:
            buckets[key] = {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "sources": case.get("sources") or [],
                "confidence": case.get("confidence"),
                "comment": case.get("comment") or "",
                "sessions": {case.get("session_id") or "unknown"},
                "count": 0,
                "first_ts": case.get("ts") or "",
            }
        item = buckets[key]
        item["count"] += 1
        item["sessions"].add(case.get("session_id") or "unknown")
        if not item["comment"] and case.get("comment"):
            item["comment"] = case["comment"]

    return list(buckets.values())


# =====================================================================
# 3. 调 DeepSeek 当判官
# =====================================================================
def build_client(api_key: str, base_url: str) -> OpenAI:
    """DeepSeek 完全兼容 OpenAI SDK，改 base_url 即可。"""
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0)


def _fmt_contexts(contexts: List[str]) -> str:
    """把上下文列表拼成 Prompt 里的证据区，带体积上限。"""
    if not contexts:
        return "（本轮没有任何检索结果／上下文为空）"

    parts: List[str] = []
    used = 0
    truncated = False

    for i, ctx in enumerate(contexts, 1):
        text = str(ctx).strip()[:MAX_CTX_PER_ITEM]
        if used + len(text) > MAX_CTX_TOTAL:
            truncated = True
            break
        parts.append(f"[上下文 {i}]\n{text}")
        used += len(text)

    body = "\n\n".join(parts)
    if truncated:
        body += "\n\n（注：上下文过多，后续块已截断）"
    return body


def _call_llm(client: OpenAI, model: str, user_prompt: str,
              temperature: float, retries: int = 3) -> str:
    """带重试 + JSON 模式降级的单次调用。"""
    use_json_mode = True
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": 900,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()

        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            # 该模型不支持 response_format → 摘掉参数重试
            if use_json_mode and ("response_format" in msg or "json_object" in msg):
                use_json_mode = False
                continue
            if attempt < retries:
                backoff = min(2 ** attempt, 10)
                print(f"   ⚠️ 第 {attempt} 次调用失败（{e}），{backoff}s 后重试…", file=sys.stderr)
                time.sleep(backoff)

    raise RuntimeError(f"DeepSeek 调用失败：{last_err}")


def _parse_json_block(text: str) -> Optional[dict]:
    """从模型输出里抠出第一个 JSON 对象（防它包 ```json 或加前言）。"""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def normalize_label(raw: str) -> str:
    """把模型给出的任意写法映射到四分类之一。

    策略：先做标签名直接匹配（含"无效反馈"这类关键子串），再退回关键词打分。
    """
    text = (raw or "").strip()
    if text in LABELS:
        return text

    # 直接包含某个标准标签名
    for label in LABELS:
        if label in text:
            return label

    scores = {label: 0 for label in LABELS}
    lower = text.lower()
    for label, kws in LABEL_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in lower:
                scores[label] += 1

    best = max(scores, key=scores.get)
    # 一个关键词都没命中：说明模型彻底跑偏。
    # 默认"生成幻觉"而不是"无效反馈"——前者会被人复核，后者会被静默丢弃（漏掉真 bug 更可怕）。
    return best if scores[best] > 0 else LABEL_HALLUCINATION


def analyze_one(client: OpenAI, model: str, case: dict, temperature: float) -> dict:
    """审计单条 bad case。永不抛异常，保证整批能跑完。"""
    question = case["question"]
    answer = case["answer"]
    contexts = case.get("contexts") or []

    user_prompt = USER_TEMPLATE.format(
        question=question or "（空）",
        ctx_count=len(contexts),
        contexts=_fmt_contexts(contexts),
        answer=answer or "（空）",
        comment=case.get("comment") or "（无）",
    )

    result = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "sources": case.get("sources") or [],
        "count": case.get("count", 1),
        "sessions": len(case.get("sessions") or []),
        "first_ts": case.get("first_ts", ""),
        "label": None,          # 四分类；None 表示调用失败、未能判定
        "degraded": False,      # True = 模型没吐合法 JSON，标签是关键词初判的，需人工复核
        "verdict": "",
        "evidence": "",
        "suggestion": "",
        "raw": "",
        "error": None,
    }

    try:
        raw = _call_llm(client, model, user_prompt, temperature)
        result["raw"] = raw

        data = _parse_json_block(raw)
        if data:
            result["label"] = normalize_label(data.get("label") or data.get("dimension") or "")
            result["verdict"] = str(data.get("verdict") or data.get("reason") or "").strip()
            result["evidence"] = str(data.get("evidence") or "").strip()
            result["suggestion"] = str(data.get("suggestion") or "").strip()
        else:
            # JSON 坏了：至少把标签用关键词捞出来，原文留档供人工复核
            result["label"] = normalize_label(raw)
            result["degraded"] = True
            result["verdict"] = (raw[:200] or "模型返回为空") + "（模型未返回合法 JSON，建议人工复核）"
            result["evidence"] = "无"
            result["suggestion"] = "（模型未返回结构化结果，这是关键词初判，建议人工复核该条）"

        # 兜底文案：保证报告每条都四项齐全
        if not result["verdict"]:
            result["verdict"] = "（模型未给出判定说明）"
        if not result["evidence"]:
            result["evidence"] = "无"
        if not result["suggestion"]:
            result["suggestion"] = (
                "忽略该反馈（回答已可在上下文中找到依据）"
                if result["label"] == LABEL_INVALID
                else "（模型未给出建议，建议人工复核）"
            )

    except Exception as e:  # noqa: BLE001
        result["error"] = str(e)
        result["label"] = None      # 明确标记"未判定"，不污染四分类统计
        result["verdict"] = f"审计调用失败：{e}"
        result["evidence"] = "无"
        result["suggestion"] = "（重跑脚本可补齐该条）"

    return result


# =====================================================================
# 4. 报告生成（严格按要求的版式）
# =====================================================================
def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …（已截断）"


def _weight(r: dict) -> int:
    """一条去重后的 case 代表多少次真实点踩。"""
    try:
        return max(1, int(r.get("count") or 1))
    except (TypeError, ValueError):
        return 1


def build_report(results: List[dict], meta: dict) -> str:
    judged = [r for r in results if r.get("label")]
    unseen = [r for r in results if not r.get("label")]

    weighted = Counter()
    for r in judged:
        weighted[r["label"]] += _weight(r)

    invalid_w = weighted.get(LABEL_INVALID, 0)
    valid_w = sum(weighted.get(lb, 0) for lb in VALID_LABELS)
    total_w = valid_w + invalid_w

    lines: List[str] = []
    add = lines.append

    # ---------- 标题 + 统计 ----------
    add("# 🚨 数据飞轮诊断报告")
    add("")
    add(f"> 数据源：`{meta['log_path']}` ｜ 裁判模型：`{meta['model']}` ｜ 生成时间：{meta['generated_at']}")
    add(f"> 原始反馈 {meta['total_records']} 条（👍 {meta['good']} / 👎 {meta['bad_raw']}）；"
        f"去重后 {len(results)} 条待审计"
        + (f"；{meta['no_context']} 条因缺少上下文被跳过" if meta.get("no_context") else "")
        + "。")
    add("")
    add(f"## 📊 归因统计：有效 Bad Case {valid_w} 条 | 主观无效反馈 {invalid_w} 条")
    add("")

    if judged:
        add("| 归因标签 | 点踩次数 | 占比 |")
        add("| --- | ---: | ---: |")
        for lb in LABELS:
            c = weighted.get(lb, 0)
            ratio = f"{c / total_w * 100:.1f}%" if total_w else "0.0%"
            add(f"| {LABEL_ICON[lb]} {lb} | {c} | {ratio} |")
        add("")
        if invalid_w and total_w:
            add(f"**降噪结论**：{invalid_w / total_w * 100:.1f}% 的点踩属于主观无效反馈，"
                f"说明真实缺陷率被用户情绪放大了。请先看下面 {valid_w} 条有效 case；"
                f"主观那部分直接忽略（建议在前端加「哪里不满意」多选，把主观差评提前筛掉）。")
        else:
            add("**降噪结论**：本轮没有发现主观无效反馈，所有点踩都能对应到系统缺陷，建议逐条跟进。")
        add("")
    else:
        add("> 本轮所有记录都未能判定（可能是 API 调用失败），请查看 `--json-out` 产物或重跑。")
        add("")

    add("---")
    add("")

    # ---------- 案例明细：有效 case 优先（次数多的先看），主观无效垫底 ----------
    ordered = sorted(
        judged,
        key=lambda r: (r["label"] == LABEL_INVALID, -_weight(r), r.get("first_ts") or ""),
    )

    for idx, r in enumerate(ordered, 1):
        icon = LABEL_ICON.get(r["label"], "•")
        degraded = " ⚠️（关键词初判，建议复核）" if r.get("degraded") else ""
        add(f"### 案例 {idx}")
        add(f"- **用户问题**：{_truncate(r['question'], 300) or '（空）'}")
        add(f"- **归因标签**：【{icon} {r['label']}】{degraded}")
        add(f"- **事实判定**：{_truncate(r['verdict'], 500)}")
        add(f"- **优化建议**：{_truncate(r['suggestion'], 400)}")
        if r.get("evidence") and r["evidence"] != "无":
            add(f"- **上下文依据**：「{_truncate(r['evidence'], 200)}」")
        add(f"- **出现次数**：{_weight(r)} 次 ｜ **涉及会话**：{r.get('sessions', 1)} 个"
            f" ｜ **首次**：{r.get('first_ts') or '未知'}"
            f" ｜ **来源**：{'、'.join(r.get('sources') or []) or '（无）'}")
        add("")

        # 证据附录：方便人工复核，不干扰正文阅读
        ctxs = r.get("contexts") or []
        add("<details><summary>📎 证据附录：检索上下文（" + str(len(ctxs)) + " 块）与 AI 回答</summary>")
        add("")
        if ctxs:
            for i, ctx in enumerate(ctxs[:6], 1):
                add(f"**上下文 {i}**")
                add("")
                add("```text")
                add(_truncate(ctx, CTX_EXCERPT_CHARS))
                add("```")
                add("")
            if len(ctxs) > 6:
                add(f"*（其余 {len(ctxs) - 6} 块上下文已省略）*")
                add("")
        else:
            add("*（本轮没有检索到任何上下文）*")
            add("")
        add("**AI 回答**")
        add("")
        add("```text")
        add(_truncate(r["answer"], ANSWER_EXCERPT_CHARS) or "（空）")
        add("```")
        add("")
        add("</details>")
        add("")

    # ---------- 未能判定 ----------
    if unseen:
        add("---")
        add("")
        add(f"## ⚠️ 未能判定（{len(unseen)} 条）")
        add("")
        add("这些记录在调用裁判模型时失败（网络/限流/额度），未计入上面的统计。重跑脚本即可补齐。")
        add("")
        for r in unseen:
            add(f"- {_truncate(r['question'], 60)} —— {_truncate(r.get('error') or '', 120)}")
        add("")

    # ---------- 附录：判定口径 ----------
    add("---")
    add("")
    add("## 📐 附：判定口径（可复核）")
    add("")
    add("| 标签 | 触发条件 | 优化落点 |")
    add("| --- | --- | --- |")
    add(f"| 🔍 {LABEL_RETRIEVAL_MISS} | 上下文为空 / 与问题无关 / 明确写着未找到 | 索引切分、检索 top_k、查询改写、混合检索权重 |")
    add(f"| 🌀 {LABEL_HALLUCINATION} | 上下文有料，但回答含上下文里没有的关键事实 | 提示词约束、temperature、生成后 groundedness 校验 |")
    add(f"| 🧩 {LABEL_FORMAT} | 乱码 / 结构损坏 / 标签泄漏 / 截断 | 结构化输出 Schema、流式清洗、max_tokens |")
    add(f"| 🙈 {LABEL_INVALID} | 关键事实都能在上下文找到出处，且正确回答了问题 | 不优化系统；优化反馈入口（增加不满意原因多选） |")
    add("")
    add(f"*本报告由 `flywheel_analyzer.py` 自动生成于 {meta['generated_at']}*")
    add("")

    return "\n".join(lines)


# =====================================================================
# 5. 主流程
# =====================================================================
def main() -> None:
    load_dotenv(BASE_DIR / ".env")   # 读取 DEEPSEEK_API_KEY

    parser = argparse.ArgumentParser(description="RAG 数据飞轮 · 带证据的 Bad Case 归因")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="反馈日志路径（JSONL）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="报告输出路径（Markdown）")
    parser.add_argument("--model", default="deepseek-chat", help="裁判模型")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="DeepSeek API 地址")
    parser.add_argument("--workers", type=int, default=3, help="并发线程数")
    parser.add_argument("--limit", type=int, default=0, help="最多审计多少条（0=全部）")
    parser.add_argument("--temperature", type=float, default=0.0, help="裁判温度；审计任务建议 0")
    parser.add_argument("--json-out", default="", help="可选：结构化结果另存 JSON")
    parser.add_argument("--include-no-context", action="store_true",
                        help="连缺少 contexts 的老数据也强行审计（不推荐：没有证据只能猜）")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("❌ 没找到 DEEPSEEK_API_KEY，请检查项目根目录的 .env", file=sys.stderr)
        sys.exit(1)

    log_path = Path(args.log).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    # ---- 1) 读数据 ----
    print(f"📂 读取反馈日志：{log_path}")
    records, good, bad, _broken = load_feedback(log_path)
    bad_cases = [r for r in records if r.get("feedback") == "bad"]
    print(f"   总记录 {len(records)} ｜ 👍 {good} ｜ 👎 {bad}")

    if not bad_cases:
        print("ℹ️ 没有 bad case，无需分析。")
        return

    # ---- 2) 证据缺失过滤（关键：没上下文就别硬判） ----
    no_ctx = []
    usable = []
    for c in bad_cases:
        if ("contexts" in c) or args.include_no_context:
            usable.append(c)          # 有字段（哪怕为空列表）就算"有证据"：空列表本身就说明检索没命中
        else:
            no_ctx.append(c)          # 字段完全不存在 = 升级前的老数据，硬判只能瞎猜

    if no_ctx:
        print(f"⚠️ {len(no_ctx)} 条缺少 contexts 字段（升级前的老数据），已跳过；"
              f"如需强行审计加 --include-no-context", file=sys.stderr)
    empty_ctx = sum(1 for c in usable if not c.get("contexts"))
    print(f"🧾 待审计 {len(usable)} 条（其中 {empty_ctx} 条上下文为空 → 大概率是检索未命中）")

    if not usable:
        print("ℹ️ 没有可审计的数据。")
        return

    cases = dedupe_bad_cases(usable)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    print(f"🧹 去重后 {len(cases)} 条（原 {len(usable)} 条）")

    # ---- 3) 并发审计 ----
    client = build_client(api_key, args.base_url)
    results: List[dict] = []
    total = len(cases)
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {
            pool.submit(analyze_one, client, args.model, case, args.temperature): case
            for case in cases
        }
        for fut in as_completed(future_map):
            done += 1
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001 - analyze_one 已兜底，这里再兜一层
                case = future_map[fut]
                res = {
                    "question": case["question"], "answer": case["answer"],
                    "contexts": case.get("contexts") or [], "sources": case.get("sources") or [],
                    "count": case.get("count", 1), "sessions": 1,
                    "first_ts": case.get("first_ts", ""), "label": None,
                    "verdict": f"审计异常：{e}", "evidence": "无",
                    "suggestion": "（请重跑）", "raw": "", "error": str(e),
                }
            results.append(res)
            tag = res["label"] or "未判定"
            print(f"   [{done}/{total}] {tag} ← {res['question'][:40]}")

    # 恢复稳定顺序，让报告可 diff
    order = {c["question"] + "\x00" + c["answer"]: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order.get(r["question"] + "\x00" + r["answer"], 10 ** 9))

    # ---- 4) 生成报告 ----
    meta = {
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "log_path": str(log_path),
        "model": args.model,
        "total_records": len(records),
        "good": good,
        "bad_raw": len(bad_cases),
        "no_context": len(no_ctx),
    }

    report = build_report(results, meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    if args.json_out:
        json_path = Path(args.json_out).expanduser().resolve()
        json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"🧾 结构化结果已保存：{json_path}")

    # ---- 5) 控制台摘要 ----
    weighted = Counter()
    for r in results:
        if r["label"]:
            weighted[r["label"]] += _weight(r)
    valid_w = sum(weighted.get(lb, 0) for lb in VALID_LABELS)
    invalid_w = weighted.get(LABEL_INVALID, 0)

    print("\n=== 归因分布（按点踩次数） ===")
    for lb in LABELS:
        c = weighted.get(lb, 0)
        bar = "█" * int(c / max(1, valid_w + invalid_w) * 30)
        print(f"  {LABEL_ICON[lb]} {lb:<16} {c:>3}  {bar}")
    print(f"\n  有效 Bad Case {valid_w} 条 ｜ 主观无效反馈 {invalid_w} 条")
    print(f"✅ 报告已生成：{out_path}")


if __name__ == "__main__":
    main()
