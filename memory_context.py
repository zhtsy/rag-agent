# -*- coding: utf-8 -*-
"""记忆相关的纯函数：会话 ID 兜底、截断、滚动摘要提示词与渲染。

为什么单独一个文件、且**不 import 任何模型/向量库**：
    这些函数是最容易被写错、也最需要测的部分（截断边界、摘要把实体名吃掉），
    如果它们长在 rag_agent.py 里，测试就得先加载 chroma + bge-reranker（十几秒、1.2GB），
    等于测不了。做法和 confidence.py / evidence.py 一样：把纯逻辑抽出来。

      1. resolve_session_id       —— 客户端不传 session_id 时怎么兜底（防串台）
      2. smart_truncate           —— 历史里的旧答案按句截断（别把话说一半）
      3. build_summary_prompt     —— 让模型把滑出窗口的旧轮次压成一段摘要
      4. ensure_summary_entities  —— 摘要漏掉实体名时，用词表兜底补回来
      5. render_summary_block     —— 把摘要渲染成可注入的文字
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Sequence

# 句末标点：截断尽量落在这里，读起来才是完整的句子
SENTENCE_ENDS = "。！？!?；;\n"
# 退而求其次的停顿标点
CLAUSE_ENDS = "，,、：:"


# ---------------------------------------------------------------------
# 历史窗口：按 provider 决定给几轮
# ---------------------------------------------------------------------
# 写死一个数字是错的，因为两条路的上下文预算差一个数量级。
# 注意"provider 名"不等于"生成用的模型"：hybrid 的**决策**走 DeepSeek，
# 但**生成**走本地 Ollama，所以它受 num_ctx 约束，不能按 128K 算。
#   · deepseek：生成也在云端，128K 上下文，16 轮 ≈ 3000 字，宽松；
#   · hybrid  ：生成走本地 7B，num_ctx=4096，8 轮 ≈ 1500 字（≈1000 token）
#                + system/证据 ≈ 1600 token，合计约 2600，留得住余量；
#   · local   ：纯本地且性能敏感，3 轮（≈570 字）最稳。
HISTORY_TURNS_BY_PROVIDER = {"deepseek": 16, "hybrid": 8, "local": 3}
DEFAULT_HISTORY_TURNS = 8


def history_turns_for(provider: str) -> int:
    """按模型档位取历史窗口轮数。

    抽成纯函数是为了能测：rag_agent 一 import 就要加载 bge + reranker（1.2GB），
    在 CI 里根本跑不起来，而"降级到本地时窗口有没有收回"恰恰是必须验证的。
    """
    return HISTORY_TURNS_BY_PROVIDER.get((provider or "").strip().lower(), DEFAULT_HISTORY_TURNS)


# ---------------------------------------------------------------------
# 会话 ID 兜底
# ---------------------------------------------------------------------
PLACEHOLDER_SESSION_IDS = {"default_session", "default", ""}


def resolve_session_id(
    session_id: str,
    client_host: str = "",
    user_agent: str = "",
    forwarded_for: str = "",
) -> str:
    """决定这次请求落到哪个会话。

    之前是 `request.session_id or "default_session"`：客户端不传就全落到同一个
    `default_session`。单机 demo 看不出问题，只要有两个使用者，A 的问题就会出现在
    B 的上下文里——这不是"记忆没做好"，是**串台**，比没有记忆严重得多。

    三档：
      1. 客户端传了真 ID（不是占位符）→ 直接用，绝大多数情况；
      2. 没传 / 传了占位符 → `anon-` + 客户端指纹，同一设备复用同一会话；
      3. 连指纹都拿不到 → 每次新会话（宁可没记忆，也不串台）。
    """
    sid = (session_id or "").strip()
    if sid not in PLACEHOLDER_SESSION_IDS:
        return sid

    seed = "|".join([client_host or "", user_agent or "", forwarded_for or ""])
    if not seed.strip("|"):
        return f"anon-{uuid.uuid4().hex[:12]}"
    return "anon-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------
# 历史压缩
# ---------------------------------------------------------------------


def smart_truncate(text: str, limit: int) -> str:
    """把文本压到 limit 字以内，尽量在句末断开。

    为什么不能直接 text[:50]：
        "代码是一只橘猫，它的主人是小李，2024 年 3 月搬到深圳"
        截 50 字能忍，截 20 字会变成 "代码是一只橘猫，它的主人是小李"——看着完整，
        但"搬到深圳"这个关键信息丢了，下一轮问"他搬去哪了"就答不上来。
        所以正确做法不是截得更短，而是**保留结论、丢掉修饰**：优先在句末断，
        断不了再在逗号处断，实在没有标点才硬切。
    """
    if not text:
        return ""
    t = str(text).strip()
    if limit <= 0:
        return ""
    if len(t) <= limit:
        return t

    head = t[:limit]

    # 1) 句末标点（且不能太靠前，否则等于没截到内容）
    cut = max(head.rfind(ch) for ch in SENTENCE_ENDS)
    if cut >= max(1, int(limit * 0.5)):
        return head[: cut + 1].strip()

    # 2) 逗号/顿号（保留逗号，表示话没说完）
    cut = max(head.rfind(ch) for ch in CLAUSE_ENDS)
    if cut >= max(1, int(limit * 0.6)):
        return head[: cut + 1].rstrip() + "…"

    # 3) 没有任何标点：硬切，用省略号明示"这里被截过"
    return head.rstrip() + "…"


# ---------------------------------------------------------------------
# 滚动摘要
# ---------------------------------------------------------------------
SUMMARY_PROMPT = """你在压缩一段较早的对话。它的原文马上要从上下文窗口里删掉，
所以这段摘要将是模型对它唯一的记忆。

必须保留（丢了就补不回来）：
1. 所有专有名称：人名、宠物名、品种名、地名、项目名、产品名，一字不改照抄
2. 所有数字与结论：年龄、数量、日期、分数、配置值
3. 用户明确提出过的要求、约束、偏好（"要用 Python""不要表格"这类）

可以丢：
- 客套话、重复表述
- 工具调用的过程（查了什么、怎么查的）
- 与后续问题无关的铺陈

要求：
- 不超过 __LIMIT__ 字
- 用陈述句，按"谁—做了什么—结论是什么"组织
- 只输出摘要正文，不要标题、不要解释、不要"好的"

【已有摘要（可能为空，请把新内容合并进去，不要丢掉其中仍然有效的信息）】
__PREV__

【本轮要合并进摘要的对话】
__TURNS__
"""


def build_summary_prompt(
    prev_summary: str,
    turns: Sequence[Dict[str, Any]],
    limit: int = 240,
) -> str:
    """拼滚动摘要的提示词。

    turns 里每项形如 {"turn_no": 3, "question": "...", "answer": "..."}。
    问答都做了截断：摘要是压缩，不是誊抄，给全文只会让模型把摘要写得跟原文一样长。

    用 replace 而不是 str.format：提示词里可能出现大括号，format 会当占位符报 KeyError。
    """
    blocks: List[str] = []
    for t in turns or []:
        no = t.get("turn_no")
        q = smart_truncate(str(t.get("question") or ""), 160)
        a = smart_truncate(str(t.get("answer") or ""), 400)
        if not q and not a:
            continue
        head = f"[第 {no} 轮]" if no is not None else "[更早的一轮]"
        blocks.append(f"{head}\n问：{q}\n答：{a}")

    return (
        SUMMARY_PROMPT
        .replace("__LIMIT__", str(limit))
        .replace("__PREV__", (prev_summary or "（无）").strip())
        .replace("__TURNS__", "\n\n".join(blocks) if blocks else "（无）")
    )


def ensure_summary_entities(
    summary: str,
    source_text: str,
    known_entities: Sequence[str],
) -> str:
    """摘要漏掉的实体名，从原始轮次里捡回来追加到末尾。

    为什么需要这道兜底：
        提示词里写十遍"保留人名"，7B/小模型仍会把它写成"主人的宠物是那只猫"。
        实体名一旦丢，下一轮"它几岁"就再也指代不了——这是摘要最致命的信息损失，
        而且是静默的（摘要读起来很通顺，没人发现名字没了）。
        所以不指望模型，直接用知识库实体词表在原文里扫一遍，缺哪个补哪个。
    """
    text = str(summary or "").strip()
    src = str(source_text or "")
    if not src:
        return text
    missing = [e for e in known_entities if e and e in src and e not in text]
    if not missing:
        return text
    return f"{text}\n【涉及实体】{'、'.join(missing)}"


def render_summary_block(summary: str, max_chars: int = 600) -> str:
    """把摘要渲染成可拼进上下文的文字。

    开头必须写清"这是背景，不是证据"：
        否则模型会拿摘要里的旧结论当本轮答案，而旧结论可能已经过时，
        或者根本是上一轮的错误答案——那比"没有记忆"更难排查。
    """
    text = str(summary or "").strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = smart_truncate(text, max_chars)
    return (
        "【更早的对话摘要】（这是本次会话中已经被压缩的历史，"
        "只用于理解指代和延续任务；它不是本轮问题的证据，"
        "与检索到的资料冲突时一律以资料为准）\n" + text
    )
