# -*- coding: utf-8 -*-
"""文本清洗：把"解析器吐出来的样子"变成"人能读的样子"。

为什么单独一层
─────────────
解析出来的脏东西和模型会不会答错是两件事，但**后果一样**：
  · 页眉页脚 / 水印被切进块里 → 每块都带一句"内部资料 第 3 页"，rerank 被这些
    高频词拉高分，真正的内容反而排后面；
  · PDF 的硬换行 → "加里奥由禁魔石制\n成" 被切成两块，两块的向量都不完整；
  · 连续空行 → 切块器把空行当段落边界，一个 200 字的段落被切成三个碎片。

这一层刻意写成**纯函数、零依赖**：能在 CI 里秒跑，也能单独拿去复用。
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

# 行尾是这些标点，说明这句没说完，下一行要接上
_NO_END = "，、；：,;:（(【[「“‘"
# 行首是这些标点，说明它是上一行的延续
_NO_START = "，、；：,;:）)】]」”’"


def _furniture_key(line: str) -> str:
    """把"会变的部分"抹掉再比较：页码、日期、空格。

    `第 1 页` / `第 2 页` 逐字比较是两个不同的行，永远凑不够阈值——
    但它们是同一个页脚。归一化之后才统计得出来。

    **只对短行归一化**。长行里出现的数字往往是内容本身
    （"第 1 季度营收 128 万" vs "第 2 季度营收 143 万"），
    归一化会把两段完全不同的正文判成同一行，然后一起删掉。
    """
    s = re.sub(r"\s+", " ", line.strip())
    if len(s) <= 20:
        s = re.sub(r"\d+", "#", s)
    return s


def strip_page_furniture(
    pages: Sequence[str],
    edge_lines: int = 2,
    ratio: float = 0.6,
) -> List[str]:
    """删掉在多页首尾重复出现的行（页眉、页脚、页码、水印）。

    判据不是"长得像页码"，而是**统计**：同一条（归一化后的）行在超过 `ratio`
    比例的页面的首 `edge_lines` 行或尾 `edge_lines` 行里出现，就当作版式噪声。

    两个细节都是实测逼出来的：
      · 页码会变（`第 1 页` vs `第 2 页`），必须先做数字归一化再统计；
      · 只有 2 页的文档也得处理 —— 2 页里出现 2 次就是 100%，同样成立。
    """
    pages = [p or "" for p in pages]
    if len(pages) < 2:  # 单页无从比较
        return list(pages)

    def edges(text: str) -> List[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[:edge_lines] + lines[-edge_lines:] if lines else []

    counter: dict[str, int] = {}
    for p in pages:
        for key in {_furniture_key(ln) for ln in edges(p)}:
            counter[key] = counter.get(key, 0) + 1

    threshold = max(2, int(len(pages) * ratio + 0.999))
    furniture = {k for k, c in counter.items() if c >= threshold and k}
    if not furniture:
        return list(pages)

    out = []
    for p in pages:
        lines = p.splitlines()
        # 非空行的位置索引：只删首尾 edge_lines 条非空行里的噪声，
        # 正文中间真出现同样的句子（比如水印字样被引用）不该被误删。
        nonempty_idx = [i for i, ln in enumerate(lines) if ln.strip()]
        head = set(nonempty_idx[:edge_lines])
        tail = set(nonempty_idx[-edge_lines:])
        edge_positions = head | tail
        kept = [
            ln
            for i, ln in enumerate(lines)
            if not (i in edge_positions and _furniture_key(ln) in furniture)
        ]
        out.append("\n".join(kept))
    return out


def unmangle_lines(text: str) -> str:
    """把 PDF 的硬换行接回去，但不破坏真正的段落。

    三条规则：
      1. 英文断词：行尾 `-` 直接粘上（`informa-\\ntion` → `information`）；
      2. 中文软换行：上一行不以句末标点结尾、下一行不以标点开头 → 接成一行；
      3. 中英混排同理，但下一行以英文大写开头时倾向是新句子，保守不接。
    """
    if not text:
        return ""
    lines = text.splitlines()
    out: List[str] = []
    for ln in lines:
        cur = ln.rstrip()
        if not out:
            out.append(cur)
            continue
        prev = out[-1]
        if not prev:                      # 上一行是空行 = 真段落边界
            out.append(cur)
            continue
        if not cur:
            out.append(cur)
            continue

        # 规则 1：英文断词
        if prev.endswith("-") and cur[:1].isalpha():
            out[-1] = prev[:-1] + cur
            continue

        # 规则 2 / 3：中文软换行
        prev_tail = prev.rstrip()[-1:]
        cur_head = cur.lstrip()[:1]
        prev_is_cjk = bool(prev_tail) and "\u4e00" <= prev_tail <= "\u9fff"
        if prev_is_cjk and prev_tail not in "。！？；.!?;：:" and cur_head not in _NO_START:
            out[-1] = prev + cur.lstrip()
            continue

        out.append(cur)
    return "\n".join(out)


def normalize_ws(text: str) -> str:
    """空白规范化：全角空格、行尾空白、3 个以上空行压成 1 个。"""
    if not text:
        return ""
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """单页/单文档的完整清洗链。顺序有讲究：先接行，再压空白。"""
    return normalize_ws(unmangle_lines(text or ""))


def clean_pages(pages: Iterable[str]) -> List[str]:
    """多页清洗：先去版式噪声（要跨页统计），再逐页清洗。"""
    pages = list(pages)
    return [clean_text(p) for p in strip_page_furniture(pages)]
