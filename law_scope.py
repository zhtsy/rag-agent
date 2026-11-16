# -*- coding: utf-8 -*-
"""law_scope.py —— 「用户点名的这部法，库里到底有没有」。

为什么单开一个纯函数模块
───────────────────────
这件事看着像一行 `if`，但它决定的是**要不要拒答**，判错的代价不对称：

  · 该拒没拒 → 拿《招标投标法》的条文回答《计量法》的问题。用户看不出来，
    因为它长得像法条、还带着来源标注。这是最坏的一种错。
  · 不该拒却拒了 → 一个本可以答的问题变成"没有收录"。用户至少知道没答。

所以判据要能单独测。这里只做字符串归一与比对，不碰索引、不碰模型，
`tests/test_law_scope.py` 在纯 pytest 环境下就能跑。

真实场景（2026-11-12）
──────────────────────
`test_cases_pdf.json` 的 P03 问「《计量法》规定国家采用什么计量单位制？」。
《计量法》那份 PDF 的字体缺 ToUnicode 映射，抽出来全是 `/G21 /G22`，
而且没有内嵌位图（OCR 也无从下手），所以它**确实不在库里**。
但检索照样返回了 5 块《招标投标法》——里面也有"计量"二字，相关性门槛过得去。
答案于是"有出处、有分数、但不是问的那部法"。
"""

from __future__ import annotations

import re
from typing import Iterable, List

# 书名号里的法名。上限 40 字是为了不把「《关于修改〈…〉的决定》」这类长标题吞掉。
_LAW_REF = re.compile(r"《([^》]{2,40})》")

# 这些不是"一部法"，是立法文件/解释。库里按篇收录时它们往往并进母法，
# 拿它们去比对必然找不到，会造成误拒答。
_SKIP_MARKERS = ("决定", "修正案", "决议", "解释", "说明")


def law_stem(name: str) -> str:
    """归一成"可以互相包含比对"的短名。

    去掉国名前缀、空白、下划线、扩展名——《中华人民共和国专利法》、
    `专利法.pdf`、`民法典_合同编.md` 都要能和用户写的《专利法》《民法典》对上。
    """
    s = (name or "").strip()
    s = re.sub(r"\.(pdf|md|txt|docx)$", "", s, flags=re.I)
    s = s.replace("中华人民共和国", "")
    s = re.sub(r"[《》〈〉\s_\-·—]", "", s)
    return s


def mentioned_laws(query: str) -> List[str]:
    """问句里点名了哪几部法（去掉立法文件类标题）。"""
    out: List[str] = []
    for name in _LAW_REF.findall(query or ""):
        if any(m in name for m in _SKIP_MARKERS):
            continue
        name = name.strip()
        if name and name not in out:
            out.append(name)
    return out


def missing_laws(query: str, source_files: Iterable[str]) -> List[str]:
    """点名了、但库里一篇都没有的法。空列表表示"不用拒答"。

    两个"宁可放过"的出口，都是为了避免误拒答：
      · 问句里没有书名号 → 返回空（"专利法第三十六条"这种不带书名号的，
        交给既有链路处理，这里不擅自拦）；
      · 拿不到来源清单（索引没起来/读失败）→ 返回空。宁可让它去检索，
        也不要在索引异常时把所有问题都拒成"没有收录"。
    """
    names = mentioned_laws(query)
    if not names:
        return []
    stems = [law_stem(f) for f in source_files if f]
    stems = [s for s in stems if s]
    if not stems:
        return []

    missing = []
    for name in names:
        stem = law_stem(name)
        if not stem:
            continue
        # 双向包含：库里是「民法典合同编」，用户写「民法典」；或反之。
        if not any(stem in s or s in stem for s in stems):
            missing.append(name)
    # 只要有一部能对上就不拒——问了《专利法》和《商标法》，答复专利法那半也算答了。
    return missing if len(missing) == len(names) else []
