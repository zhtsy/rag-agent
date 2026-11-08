# -*- coding: utf-8 -*-
"""工具决策的兜底判据 —— 纯函数、零依赖、可离线单测。

为什么需要它
────────────
`_run_tool_decision` 末尾有一段：

    if not tool_calls:
        logger.info("📝 决策结果：不需要检索")
        return None

这一个条件同时覆盖了三种完全不同的情况：

  1. 模型**明确**判断不用查（提示词要求输出 `{"need_tool": false}`）
  2. 模型**什么都没说**（空内容 / 只吐了半句 / 输出被截断）
  3. 模型把工具调用写成正文，且正文没能还原成 tool_calls

第 1 种是判断，第 2、3 种是沉默。把沉默当判断，等于
**「我没决定」被读成「我决定不查」**。

2026-11-07 实测抓到一次：三轮评测里 T038「根据知识库，谁住在广州？」
第三轮整题返回「无法从知识库中找到相关信息。」，`status=out_of_scope`、
`confidence=None`、`sources=[]`——与一次真正的库外拒答在接口上完全一致。
日志证据：`stages={"decide":728.5,"llm":1046.3}`，没有 `graph`、没有 `retrieve`。
复现 53 次全过，说明是低频偶发（约 1%），不是某道题的固有属性。

为什么 T038 会漏到这条路
──────────────────────
`agent_node` 的 1.1 分支只要问题里点得出已知实体，就**代码直接伪造检索、不问模型**。
反向查询问的是「谁」，正确答案里的人名不在问题里，于是 1.1 覆盖不到，
只能落到 1.2 让模型决策 —— 这是 38 题里唯一一道走这条路的题。
所以修法不是加一个 try/except，而是把「用户已经明确要求查库」这件事也变成确定性逻辑。

三条规则
───────
1. **沉默必须兜底**：只有内容能解析成 `{"need_tool": false}` 才算模型说"不用查"。
   空内容、乱码、自然语言解释，一律当沉默 → 强制走一次检索。
   检索一次 3–4 秒；静默答错的代价比这大得多。
2. **用户显式要求查库时，模型无权决定不查**："根据知识库…"已经是用户的指令，
   此时哪怕模型明确输出 need_tool=false 也要查。查了没有，才是 out_of_scope。
3. **不认自然语言否定**：模型说"这不需要查询知识库"不算数，只认严格 JSON。
   宁可多跑一次检索，也不要把一句含糊的自然语言当成授权。

第 3 条有代价（寒暄可能多花一次检索），所以兜底时会打 WARNING，
观察一段时间再决定要不要放宽——放宽之前必须先有日志证明它真的常发生。
"""

import json
import re

# 用户显式要求查库的说法。刻意只收**指令式**说法：
# "知识库里有什么"是在问知识库本身，不该算指令，所以不在表里。
RETRIEVAL_PHRASES = (
    "根据知识库",
    "根据资料",
    "根据文档",
    "根据上传",
    "按照知识库",
    "依据知识库",
    "查一下知识库",
    "查知识库",
    "从知识库",
    "以知识库为准",
)

# 前缀：用来在伪造检索词时去掉它（检索词带"根据知识库，"会污染 BM25 分词）
#
# 不能写成"匹配到"知识库"就停"——"按照知识库里写的，小刘多大了"会剥成
# "里写的，小刘多大了"，检索词里带着"里写的"三个字去查 BM25。
# 改成"从动词到第一个分隔符"，并限长（超长说明这不是一个前缀短语，
# 宁可原样返回也不要吞掉正文）。
_PREFIX_RE = re.compile(r"^\s*(?:请)?(?:根据|按照|依据)[^，,：:。？?！!]{0,16}[，,：:]\s*")

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_NEED_TOOL_FALSE_RE = re.compile(r'\{\s*"need_tool"\s*:\s*false\s*\}', re.IGNORECASE)


def requires_retrieval(query: str) -> bool:
    """用户这句话是否显式要求查本地知识库。

    只认指令式说法（见 RETRIEVAL_PHRASES）。命中时调用方应跳过模型决策，
    直接注入检索请求——这不是优化，是把"用户要求"和"模型判断"分开。
    """
    q = (query or "").strip()
    if not q:
        return False
    return any(p in q for p in RETRIEVAL_PHRASES)


def strip_retrieval_prefix(query: str) -> str:
    """去掉"根据知识库，"这类前缀，剩下的才是检索词。

    只去开头一处；中间出现的不动（"小智说根据知识库…"这种要保持原样）。
    """
    q = (query or "").strip()
    if not q:
        return ""
    stripped = _PREFIX_RE.sub("", q, count=1).strip()
    return stripped or q


def explicit_no_tool(content) -> bool:
    """内容是否**明确**表达了 `need_tool: false`。

    只认严格 JSON（允许包在 markdown 代码块里）。
    自然语言否定不认——理由是这类文本太容易和别的东西混淆，
    而多跑一次检索的代价远低于把一句含糊的话当成授权。
    """
    if not isinstance(content, str):
        return False
    text = content.strip()
    if not text:
        return False

    text = _FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(text)
    except ValueError:
        # 内容里夹了一个完整对象（"好的。{"need_tool": false}"）
        return bool(_NEED_TOOL_FALSE_RE.search(text))

    if isinstance(obj, dict) and obj.get("need_tool") is False:
        return True
    return False


def should_force_retrieval(query: str, content) -> bool:
    """决策阶段没给出 tool_call 时，是否应该兜底强制检索。

    返回 True 的三种情况：
      · 用户显式要求查库（规则 2，优先级最高，模型说什么都不算）
      · 模型沉默（规则 1）
      · 模型只说了自然语言，没有给出明确结论（规则 1 + 3）

    返回 False 只有一种：模型给了严格 `{"need_tool": false}`，且用户没要求查库。
    """
    if requires_retrieval(query):
        return True
    return not explicit_no_tool(content)
