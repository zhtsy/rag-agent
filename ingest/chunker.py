# -*- coding: utf-8 -*-
"""结构感知切块：按标题分段、表格整块保留、切块时记住"这段属于哪一节"。

为什么不直接用 RecursiveCharacterTextSplitter
────────────────────────────────────────────
它对纯文本没问题，问题出在两处，而且都是实测踩到的：

1. **表格被切碎**。一张 5 行的表切成 "| 菜名 | 口味 |" 和 "| 开水白菜 | 清" 两块，
   第二块单看毫无意义，但它的向量会跟"开水白菜什么口味"高度相似——于是检索
   命中它，答案却缺了表头。
2. **切完丢层级**。200 字一块切下去，"（补充）招牌菜还有…"这种块拿不到它属于
   哪一节，追问时无法判断它讲的是川菜还是粤菜，出处也说不清"第几节"。

这里用"先切结构、再按预算装箱"两步：结构边界（标题/空行/表格）永远优先，
预算只决定在哪断开。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from .loaders import Doc

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

# 「第一条」「第三十六条」「第一百二十条之一」——法条类文档的硬边界
_ARTICLE_RE = re.compile(
    r"^第[〇零一二三四五六七八九十百千两\d]{1,12}条(?:之[〇零一二三四五六七八九十]{1,3})?(?=[\s　]|$)"
)


def _article_of(text: str) -> str:
    """段落是否以条号开头；是则返回「第三十六条」，否则空串。"""
    first = (text or "").lstrip("\n").split("\n", 1)[0].strip()
    m = _ARTICLE_RE.match(first)
    return m.group(0) if m else ""


def _blocks(text: str) -> List[Tuple[str, int, str]]:
    """把文本切成结构块：(kind, level, content)。kind ∈ heading/table/para。"""
    blocks: List[Tuple[str, int, str]] = []
    buf: List[str] = []

    def flush() -> None:
        if buf:
            body = "\n".join(buf).strip()
            if body:
                blocks.append(("para", 0, body))
            buf.clear()

    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            blocks.append(("heading", len(m.group(1)), m.group(2)))
            i += 1
            continue
        if line.strip().startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].rstrip())
                i += 1
            blocks.append(("table", 0, "\n".join(rows)))
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        buf.append(line)
        i += 1
    flush()
    return blocks


def _split_table(table: str, size: int) -> List[str]:
    """表格超预算时按行切，但**每块都重复表头**，否则后半张表读不懂。"""
    rows = [r for r in table.splitlines() if r.strip()]
    if len(rows) <= 2 or sum(len(r) for r in rows) <= size:
        return [table]

    def _is_sep(row: str) -> bool:
        body = row.replace("|", "").strip()
        return bool(body) and set(body) <= set("-: ")

    # 表头 = 第一行；紧跟一行 `| --- | --- |` 时，两行都算表头
    if len(rows) >= 2 and _is_sep(rows[1]):
        header, data = rows[:2], rows[2:]
    else:
        header, data = [], rows
    head_text = "\n".join(header)
    budget = max(size - len(head_text), size // 2)

    out, cur, cur_len = [], [], len(head_text)
    for r in data:
        if cur and cur_len + len(r) > budget:
            out.append("\n".join(([head_text] if head_text else []) + cur))
            cur, cur_len = [], len(head_text)
        cur.append(r)
        cur_len += len(r) + 1
    if cur:
        out.append("\n".join(([head_text] if head_text else []) + cur))
    return out or [table]


def _with_heading(body: str, heading: str) -> str:
    """把所在章节标题写进正文第一行。

    为什么不能只留在 metadata 里：一条法条正文是「第三十六条　发明专利的申请人
    请求实质审查的时候…」，光看这段既不知道是哪部法，也不知道是第几章。
    检索时更糟——52 部法律里都有「第三十六条」，纯向量/BM25 无法区分，
    实测「《专利法》第三十六条」会召回商标法、刑法的同号条文。
    拼进正文后，两个检索器和生成模型都能看到法名，这是廉价的上下文补全。
    """
    h = (heading or "").strip()
    return f"{h}\n{body}" if h else body


def chunk_doc(doc: Doc, size: int = 200, overlap: int = 40) -> List[Doc]:
    """单个 Doc → 若干块。每块 metadata 带 heading_path / is_table / article。

    两条路：
    * **法条类**（正文里连续出现 ≥5 个「第 N 条」）：条号是硬边界，一条一块。
    * 其余文档：沿用「先切结构、再按预算装箱」。

    为什么要给法条单开一条路
    ───────────────────────
    定长 200 字装箱会把相邻两条塞进同一块。实测《专利法》第二条回答「发明是指
    什么」时，块里前半是第一条（立法目的）、后半才是定义，向量被前者稀释，
    定义式题 12 条只拿 0.333。这不是检索坏了，是切块没有尊重文档的语义单元——
    法条自带编号，本来就有明确边界，不该按字数切断。

    处理超长条时，续块必须重新带上「法名 > 章 > 条」：否则第二块只剩「（二）…」
    这样的枚举项，既不知道是哪部法，也不知道是第几条，等于把一块废料塞进索引。
    """
    blocks = _blocks(doc.text)
    article_mode = sum(
        1 for kind, _, content in blocks if kind == "para" and _article_of(content)
    ) >= 5
    out: List[Doc] = []
    stack: List[Tuple[int, str]] = []   # (level, title)
    buf: List[str] = []
    buf_len = 0
    buf_heading = ""
    cur_article = ""

    def _prefix() -> str:
        if not cur_article:
            return buf_heading
        return f"{buf_heading} > {cur_article}" if buf_heading else cur_article

    def flush(carry: bool = True) -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        body = "\n".join(buf).strip()
        if body:
            meta = dict(doc.metadata)
            meta["heading_path"] = buf_heading
            meta["is_table"] = False
            if cur_article:
                meta["article"] = cur_article
            out.append(Doc(_with_heading(body, _prefix()), meta))
        if carry and overlap > 0 and len(body) > overlap:
            tail = body[-overlap:]
            # 从标点后开始，避免把半句话当重叠
            cut = max(tail.rfind(p) for p in "。！？；\n")
            tail = tail[cut + 1:] if cut > 0 else tail
            buf = [tail] if tail.strip() else []
            buf_len = len(tail)
        else:
            buf, buf_len = [], 0

    for kind, level, content in blocks:
        if kind == "heading":
            flush(carry=False)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, content))
            buf_heading = " > ".join(t for _, t in stack)
            cur_article = ""            # 换章了，条号前缀跟着失效
            continue

        if kind == "table":
            flush(carry=False)
            meta = dict(doc.metadata)
            meta["heading_path"] = buf_heading
            meta["is_table"] = True
            if cur_article:
                meta["article"] = cur_article
            pieces = _split_table(content, size)
            for idx, piece in enumerate(pieces, start=1):
                m = dict(meta)
                if len(pieces) > 1:
                    m["table_part"] = f"{idx}/{len(pieces)}"
                out.append(Doc(_with_heading(piece, _prefix()), m))
            continue

        if article_mode:
            art = _article_of(content)
            if art:                     # 新的一条开始：上一条到此为止
                flush(carry=False)
                cur_article = art

        # 普通段落：超预算就先切句，再装箱
        if buf_len + len(content) + 1 > size and buf:
            flush(carry=not article_mode)
        if len(content) > size:
            for sent in re.split(r"(?<=[。！？；\n])", content):
                if not sent.strip():
                    continue
                if buf_len + len(sent) > size and buf:
                    flush(carry=not article_mode)
                buf.append(sent.strip())
                buf_len += len(sent)
            continue
        buf.append(content)
        buf_len += len(content) + 1

    flush(carry=False)
    return out


def chunk_docs(docs: Sequence[Doc], size: int = 200, overlap: int = 40) -> List[Doc]:
    out: List[Doc] = []
    for d in docs:
        if not d:
            continue  # 空页不产块，但它的 metadata 已经由 loaders 标出来了
        out.extend(chunk_doc(d, size=size, overlap=overlap))
    return out


def stats(chunks: Sequence[Doc]) -> Dict[str, int]:
    return {
        "chunks": len(chunks),
        "chars": sum(len(c.text) for c in chunks),
        "tables": sum(1 for c in chunks if c.metadata.get("is_table")),
        "with_heading": sum(1 for c in chunks if c.metadata.get("heading_path")),
    }
