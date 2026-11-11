# -*- coding: utf-8 -*-
"""记忆相关的纯函数：截断、渲染、抽取提示词、解析抽取结果。

为什么单独一个文件、且**不 import 任何模型/向量库**：
    这些函数是最容易被写错、也最需要测的部分（截断边界、LLM 吐出来的脏 JSON），
    如果它们长在 rag_agent.py 里，测试就得先加载 chroma + bge-reranker（十几秒、1.2GB），
    等于测不了。做法和 confidence.py / evidence.py 一样：把纯逻辑抽出来。

      1. smart_truncate          —— 历史里的旧答案按句截断（别把话说一半）
      2. render_long_term_block  —— 把用户长期记忆拼成一段可注入 system 的文字
      3. build_extract_prompt    —— 让模型从一轮问答里抽"关于用户本人"的稳定事实
      4. parse_facts             —— 容忍 ```json 围栏、前后废话、单对象、超长字段
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any, Dict, List, Sequence

# 句末标点：截断尽量落在这里，读起来才是完整的句子
SENTENCE_ENDS = "。！？!?；;\n"
# 退而求其次的停顿标点
CLAUSE_ENDS = "，,、：:"


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


def render_long_term_block(memories: Sequence[Dict[str, Any]], max_chars: int = 320) -> str:
    """把长期记忆渲染成一段可拼进 system prompt 的文字。

    两个刻意的设计：
      · 开头写清"只作背景、不能替代本轮资料"——否则模型会拿用户画像当证据答题，
        那是比"没有记忆"更严重的错误（幻觉换了个来源）。
      · 总长度硬上限。长期记忆是**辅助**，一旦长到和检索资料抢位置，
        就等于拿"用户上次说过的话"去压"知识库里的原文"。
    """
    if not memories:
        return ""

    lines: List[str] = []
    used = 0
    for m in memories:
        key = str(m.get("key") or "").strip()
        value = str(m.get("value") or "").strip()
        if not key or not value:
            continue
        line = f"- {key}：{value}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1  # +1 是换行

    if not lines:
        return ""

    return (
        "【已知用户信息】（来自此前与该用户的对话，只作称呼和背景参考；"
        "它不是本轮问题的证据，不能替代下面的检索资料，与资料冲突时一律以资料为准）\n"
        + "\n".join(lines)
    )


# ---------------------------------------------------------------------
# 长期记忆抽取
# ---------------------------------------------------------------------
EXTRACT_PROMPT = """你是一个记忆抽取器。下面是一轮"用户提问 + 助手回答"。
请只抽出**关于用户本人**、并且**换个会话仍然成立**的信息。

【最重要的规则：只能相信用户自己说的话】
每一条事实都必须在【用户提问】里找得到依据。
助手回答里出现的人名、宠物名、地址、属性，一律不要抽——那通常是知识库里的内容，
或者是助手提到的第三方（例：助手说"代码的主人是小李"，"小李"不是你，不许记成用户称呼）。

可以抽的四类（都要求出自用户提问）：
1. 身份/称呼：姓名、职业、所在城市
2. 稳定偏好：口味、常用技术栈、喜欢的风格
3. 长期目标或约束：在准备什么、有什么限制条件
4. 反复出现的关注点：明显在连续研究同一个主题

不要抽：
- 知识库里的知识（"番茄炒蛋怎么做"不是用户信息）
- 一次性的状态（"今天有点累"）
- 本轮问题的原文
- 你推测出来的东西

没有可抽的内容就只输出 []。

输出：JSON 数组，最多 3 条，每条形如
[{"key": "称呼", "value": "小王", "category": "profile", "confidence": 0.9}]

key 不超过 12 字，value 不超过 40 字且必须是陈述片段，不要输出任何解释。

【用户提问】__Q__
【助手回答（仅用于理解上下文，不可作为信息来源）】__A__
"""


def build_extract_prompt(question: str, answer: str, answer_limit: int = 600) -> str:
    """拼抽取提示词。

    用 replace 而不是 str.format：提示词里带 JSON 示例的 {}，
    format 会把它们当占位符，一跑就 KeyError（这个坑在 mini_agent 里踩过一次）。
    回答也截断——抽取只需要语义，不需要全文，省一半 token。
    """
    return (
        EXTRACT_PROMPT
        .replace("__Q__", smart_truncate(question, 200))
        .replace("__A__", smart_truncate(answer, answer_limit))
    )


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_ARRAY_RE = re.compile(r"\[.*\]", re.S)
_OBJECT_RE = re.compile(r"\{.*\}", re.S)

MAX_FACTS = 3
MAX_KEY_CHARS = 16
MAX_VALUE_CHARS = 60
_CATEGORIES = {"profile", "preference", "goal", "focus"}
# 事实里有多少字符片段能在"用户原话"里找到，才算有依据。
# 0.5 是实测出来的：严格 1.0 会把"吃饭偏辣"这种轻微改写的正确事实判死；
# 0.3 又会让"代码的主人是小李"这种知识库内容混进来（实测 support≈0.2）。
MIN_EVIDENCE_SUPPORT = 0.5

_PUNCT_RE = re.compile(r"[\s，。！？、,.!?:：；;\"'“”‘’（）()【】\[\]—\-~]")


def _bigrams(text: str) -> set:
    """字符 bigram 集合。中文没有空格切词，用 bigram 做"依据比对"最稳。"""
    t = _PUNCT_RE.sub("", str(text or ""))
    if len(t) < 2:
        return {t} if t else set()
    return {t[i:i + 2] for i in range(len(t) - 1)}


def evidence_support(value: str, evidence: str) -> float:
    """value 的字符片段有多大比例出现在 evidence 里（0~1）。

    这条检查是给抽取结果上的"事实约束"：模型很容易把助手回答里的
    "代码的主人是小李"抽成"称呼=小李"，而用户从头到尾没说过自己叫小李。
    落库前用用户原话卡一道，比在提示词里喊十遍"只抽用户信息"都管用。
    """
    vb = _bigrams(value)
    if not vb:
        return 0.0
    eb = _bigrams(evidence)
    if not eb:
        return 0.0
    return len(vb & eb) / len(vb)


def _coerce_list(raw: Any) -> List[Any]:
    """把模型可能返回的各种形状，统一成一个 list。"""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # 有时会包一层 {"facts": [...]} / {"memories": [...]}
        for v in raw.values():
            if isinstance(v, list):
                return v
        return [raw]
    return []


def parse_facts(raw: str, evidence: str = "") -> List[Dict[str, Any]]:
    """解析抽取结果。对模型的各种不听话都要容忍，但输出必须是干净的。

    见过的坏形状：```json 围栏 / 前面写"好的，我抽取到：" / 返回单个对象而不是数组 /
    key 写成英文 / value 里塞了 200 字 / confidence 给成 "0.9" 字符串。

    evidence 传用户原话时，会额外做一道"依据检查"：value 的字符片段必须能在原话里找到
    （阈值 MIN_EVIDENCE_SUPPORT）。不传就跳过检查（纯解析场景，测试更方便）。
    """
    if not raw or not str(raw).strip():
        return []

    text = str(raw).strip()

    # 1) 剥 ``` 围栏
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    # 2) 抠 JSON（优先数组，其次单对象）
    data: Any = None
    m = _ARRAY_RE.search(text)
    if m:
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            data = None
    if data is None:
        m = _OBJECT_RE.search(text)
        if m:
            try:
                data = json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                data = None
    if data is None:
        return []

    facts: List[Dict[str, Any]] = []
    for item in _coerce_list(data):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or item.get("name") or "").strip()
        value = str(item.get("value") or item.get("content") or "").strip()
        if not key or not value:
            continue
        # 空值/占位符直接丢：模型偶尔会写 "未知" "无"
        if value in {"未知", "无", "null", "None", "-"}:
            continue
        if len(key) > MAX_KEY_CHARS or len(value) > MAX_VALUE_CHARS:
            continue
        # 依据检查：只在用户原话里出现过的内容才允许写进长期记忆
        if evidence and evidence_support(value, evidence) < MIN_EVIDENCE_SUPPORT:
            continue

        category = str(item.get("category") or "").strip().lower()
        if category not in _CATEGORIES:
            category = "profile"

        try:
            conf = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            conf = 0.6
        conf = min(max(conf, 0.1), 1.0)

        facts.append({
            "key": key,
            "value": value,
            "category": category,
            "confidence": conf,
        })
        if len(facts) >= MAX_FACTS:
            break

    return facts
