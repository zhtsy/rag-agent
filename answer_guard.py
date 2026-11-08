# -*- coding: utf-8 -*-
"""answer_guard.py —— 答案完整性兜底（确定性，只依赖检索原文）

为什么需要它
────────────
答案提示词里早就有「铁律四：限定信息不许丢」，例子正是"一只叫布丁的仓鼠"。
但生成走的是 DeepSeek 服务端采样，即便 `temperature=0.0` 也不是逐位确定的
（实测同一问题 3 次里有 1 次只答"布丁"、把"仓鼠"丢了）。
提示词已经写无可写 —— 继续加字只会稀释其他规则，所以改成**生成后做确定性校验**。

它做什么
────────
认两种句式，都只在"答案提了名字却漏了类别"时才出手：

1. 资料原文「叫X的Y / 名字叫X的Y」→ 补「（补充：X是Y。）」
2. 图谱结果「X（Y）」（本项目的图谱格式）→ 同上；
   但属性对超过 3 个就当作"枚举列表"，一律不补，避免把没问到的实体刷到答案尾巴上。

为什么不做得更激进
──────────────────
- 只认"叫X的Y"这一种句式，不做泛化的属性推断（"X在Y工作"这类关系太容易误判）；
- 补的内容**一定来自检索原文**，不是模型生成，所以不会引入幻觉；
- 答案里已经出现 Y 就一个字都不加（不做无谓改写）；
- 最多补 2 条，且同一 (X, Y) 只补一次；
- 有开关：环境变量 `RAG_ANSWER_GUARD=0` 可整体关掉。

它不是"改写模型的话"，而是"在模型漏掉原文限定成分时，把原文那句补在末尾"，
所以即使判错了，最坏结果也只是多一句原文里本来就有的事实。
"""

import os
import re

# 名字：1~12 个非标点非空白字符；类型：1~10 个中英文数字字符
_PATTERN = re.compile(
    r"(?:名字叫做|名字叫|名叫做|名叫|名为|叫做|叫)"
    r"\s*[“\"'‘「《]?"
    r"([^“”\"'‘’「」《》，。、,;；:：\s]{1,12})"
    r"[”\"'’」》]?"
    r"\s*的\s*"
    r"([\u4e00-\u9fa5A-Za-z0-9]{1,10})"
)

# 图谱检索结果自带"名字（属性）"格式，例如「小张养了布丁（仓鼠）」。
# 只按"叫X的Y"匹配的话，图谱路径的上下文永远命中不了，所以再补一条规则。
# 名字里排除虚词（的/了/是/在/有/和/与）：否则"小张养了布丁（仓鼠）"会把
# 整个"小张养了布丁"当成名字，补出"小张养了布丁是仓鼠"这种病句。
_PAREN = re.compile(r"([^\s（），。；、：:【】的了是在有和与]{1,12})（([^（）\n]{1,24})）")

# 属性对超过这个数量就认为"这是个枚举列表而不是一条事实"，不再补，
# 否则会把答案里没问到的实体一股脑补上去，变成噪音。
_MAX_PAREN_PAIRS = 3

# 「叫小明的**那个人**」「叫X的**时候**」——这些不是实体类型，补了反而错
_TYPE_BLOCKLIST = frozenset(
    {
        "那个人", "这个人", "那人", "这人", "人", "事", "物", "地",
        "时候", "地方", "东西", "名字", "方式", "类型", "问题", "情况",
        "朋友", "同事", "同学", "老师", "领导", "客户", "家伙",
    }
)

# 「叫X的**那只**」「叫X的**这个人**」：以指示代词开头的都是回指，不是类别。
# 不加这条，实测会把"名字叫'代码'的那只"补成"代码是那只"。
_DEMONSTRATIVE_PREFIX = ("那", "这", "其", "哪", "此")

# 类别不该以虚词/动词结尾（"那个人在旁边""他说的"），否则说明截取跑偏了
_BAD_TAIL = ("的", "了", "着", "是", "在", "吗", "呢", "吧", "啊")


def _is_usable_kind(kind: str) -> bool:
    if not kind or kind in _TYPE_BLOCKLIST:
        return False
    if kind.startswith(_DEMONSTRATIVE_PREFIX):
        return False
    if kind.endswith(_BAD_TAIL):
        return False
    return True


def enabled() -> bool:
    return (os.environ.get("RAG_ANSWER_GUARD", "1") or "1").strip() not in ("0", "false", "False", "off")


def completeness_supplement(answer: str, contexts, max_items: int = 2) -> str:
    """返回要追加的补充句（可能为空串）。

    参数
    ----
    answer   : 模型已经吐出的完整正文
    contexts : 本轮检索到的原始文档块列表
    """
    if not enabled() or not answer or not contexts:
        return ""

    found = []
    seen = set()

    def _consider(name: str, kind: str) -> None:
        if not name or not kind:
            return
        if not _is_usable_kind(kind):
            return
        if name not in answer:
            # 答案压根没提这个实体，不归本校验管
            return
        if kind in answer:
            # 限定成分已经在答案里了，不重复
            return
        key = (name, kind)
        if key in seen:
            return
        seen.add(key)
        found.append(key)

    # 规则一：资料原文「叫X的Y」
    for ctx in contexts:
        if not isinstance(ctx, str):
            continue
        for m in _PATTERN.finditer(ctx):
            _consider(m.group(1).strip(), m.group(2).strip())

    # 规则二：图谱结果「X（Y）」——数量多说明是枚举列表，放弃，避免刷屏
    if not found:
        pairs = []
        for ctx in contexts:
            if not isinstance(ctx, str):
                continue
            for m in _PAREN.finditer(ctx):
                name, inner = m.group(1).strip(), m.group(2).strip()
                for attr in re.split(r"[，,、；;]", inner):
                    attr = attr.strip()
                    if attr:
                        pairs.append((name, attr))
        if 0 < len(pairs) <= _MAX_PAREN_PAIRS:
            for name, attr in pairs:
                _consider(name, attr)

    if not found:
        return ""

    parts = [f"{name}是{kind}" for name, kind in found[:max_items]]
    return "（补充：" + "；".join(parts) + "。）"
