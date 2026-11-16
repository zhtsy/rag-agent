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

# 行尾是这些标点，说明这句没说完，下一行要接上。
# `《` `〈` 是法律文档特有的：法名常被排版拆成「《\n关于修改〈\n中华人民共和国专利法〉的决定》」。
_NO_END = "，、；：,;:（(【[「“‘《〈<"
# 行首是这些标点，说明它是上一行的延续
_NO_START = "，、；：,;:）)】]」”’"
# 只有句末标点才算"这段说完了"。分号/冒号/逗号都不算——PDF 里它们是排版换行点，
# 不是语义断点，接不上就会把一个法条切成五六块。
_SENTENCE_END = "。！？!?"

# 全角数字/字母 → 半角。只映射这两类，**不碰标点**：用 NFKC 会把中文逗号
# 「，」变成「,」，那会同时污染切块的句子边界和 must_contain 关键词匹配。
_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９％"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "0123456789%"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz",
)

# 竖排（旋转 90°）表格里的括号。`︵`是横排 `（` 的竖排形态，字形一样、码位不同。
_VERT_PUNCT = str.maketrans("︵︶︹︺", "（）〔〕")


def normalize_fullwidth_alnum(text: str) -> str:
    """全角数字/字母 → 半角（含"数字中间的全角小数点"）。

    政府公报 PDF 的日期常被排成「１９８４」并被拆成独立行，不归一化的话
    既拼不回整句，也匹配不上「1984」这类关键词（BM25 里是两个不同的 token）。

    `．`（U+FF0E）单独处理：它不在 `_FULLWIDTH` 表里，但税法附表满页都是
    「１．０升」「２．５吨」，译完是 `1．0` —— 数字中间夹一个全角点，
    BM25 会把 `1` 和 `0` 切成两个 token，「1.6 升排量」这类查询直接落空。
    **只在两侧都是数字时替换**：全角点也用来做省略号/间隔号，那类不该动。

    竖排括号（`︵︶︹︺`，U+FE35 区）也一并转成横排括号：它们只在旋转 90°
    排版的表格里出现，检索端不会有人输入竖排括号，留着就是永远匹配不上的死字符。

    `％`（U+FF05）同理：税率表里满页「按照货车税额的 50％计算」，而用户会输入
    「50%」。**只映射百分号这一种全角标点**，不去做全量 NFKC —— 那会把中文逗号
    句号也变成半角，直接破坏切块的句子边界和 must_contain 匹配。
    """
    out = (text or "").translate(_FULLWIDTH).translate(_VERT_PUNCT)
    return re.sub(r"(?<=\d)．(?=\d)", ".", out)


def collapse_cjk_spaces(text: str) -> str:
    """删掉「中 文 之 间」的空格。

    PDF 按字定位排版时会在字与字之间插空格（「委 托 所 完 成」）。
    **只删两侧都是中文的空格**：英文单词之间的空格是语义，不能动。
    """
    return re.sub(r"(?<=[\u4e00-\u9fff])[ \t\u3000]+(?=[\u4e00-\u9fff])", "", text or "")


def _is_fragment(line: str) -> bool:
    """这一行是不是"排版碎片"：极短，或是短的无中文行（数字/标点/少量拉丁字母）。

    判据刻意宽松：宁可多接几行，也不能让「（」「1984」「年3」各占一行。
    两个边界都是被测试钉住的：
      · 真正的中文标题（"第一章　总则"）长度超过 3 且含汉字 → 不算碎片；
      · 英文正文行（"tion retrieval"）虽然无中文，但长度远超 6 → 不算碎片，
        否则会抢在"英文断词"规则前面把它接成 `informa-tion`。
    """
    s = (line or "").strip()
    if not s:
        return False
    if len(s) <= 3:
        return True
    return len(s) <= 6 and re.search(r"[\u4e00-\u9fff]", s) is None


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

    规则（顺序有讲究，前四条是真实法律 PDF 逼出来的）：
      0. **碎片行**：极短行、或整行只有数字/标点（不含中文）→ 无条件黏上。
         政府公报里「（」「１９８４」「年３」「月１２」各占一行，不接就永远
         拼不成「（１９８４年３月１２日」。
      1. **纯标点行**："," "；" 单独占一行 → 黏到上一行末尾。原来「下一行以
         标点开头就不接」的写法在这里恰好失效：标点自己就是一行，永远接不上，
         于是每个逗号都变成一次段落分割。
      2. 英文断词：行尾 `-` 直接粘上（`informa-\\ntion` → `information`）；
      3. 中文软换行：上一行不以句末标点结尾、下一行不以标点开头 → 接成一行；
      4. 中英混排同理，但下一行以英文大写开头时倾向是新句子，保守不接。
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

        prev_tail = prev.rstrip()[-1:]
        cur_head = cur.lstrip()[:1]

        # 规则 0a：当前行是碎片 → 接（但别把句号接没了）
        if _is_fragment(cur) and prev_tail not in "。！？":
            out[-1] = prev + cur.lstrip()
            continue
        # 规则 0b：上一行是碎片 → 接
        if _is_fragment(prev) and cur_head not in "。！？":
            out[-1] = prev + cur.lstrip()
            continue
        # 规则 0c：上一行以数字收尾、当前行以汉字开头（日期被拆开的典型形态）
        if prev_tail.isdigit() and "\u4e00" <= cur_head <= "\u9fff":
            out[-1] = prev + cur.lstrip()
            continue

        # 规则 1：英文断词
        if prev.endswith("-") and cur[:1].isalpha():
            out[-1] = prev[:-1] + cur
            continue

        # 规则 2：上一行以"未说完"的标点收尾 → 接（逗号、分号、开引号都算）
        if prev_tail in _NO_END:
            out[-1] = prev + cur.lstrip()
            continue

        # 规则 3：中文软换行（上一行不是句末标点 + 当前行不是标点开头）
        prev_is_cjk = bool(prev_tail) and "\u4e00" <= prev_tail <= "\u9fff"
        if prev_is_cjk and prev_tail not in _SENTENCE_END and cur_head not in _NO_START:
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
    """单页/单文档的完整清洗链。顺序有讲究。

    ① 全角数字→半角 ② 去中文字间空格 ③ 接行 ④ 压空白。
    ①② 必须在 ③ 之前：不先归一化，「１９８４」和「年３」看起来都不是碎片，
    接行规则判断不出来；不去空格，「委 托」会让碎片行永远是碎片。
    """
    text = normalize_fullwidth_alnum(text)
    text = collapse_cjk_spaces(text)
    return normalize_ws(unmangle_lines(text))


def cjk_ratio(text: str) -> float:
    """汉字占非空白字符的比例。用来判断一段"像不像中文正文"。

    为什么需要：tesseract 在横排小字上会把中文认成拉丁字母，产出

        `AM am, (exo maa, mE.`      `BAM me IES Al FSM:`

    字数、标点都正常，只有汉字是 0 —— 单看长度完全看不出问题。这类碎片的
    汉字占比接近 0，而正常法条在 0.8 以上，中间几乎不重叠（实测 41 块图片
    块里，低于 0.3 的只有 2 块，且都是这种噪声）。
    """
    body = re.sub(r"\s+", "", text or "")
    if not body:
        return 0.0
    return len(re.findall(r"[\u4e00-\u9fff]", body)) / len(body)


def drop_garbage_lines(text: str, min_ratio: float = 0.30, min_len: int = 8,
                       min_latin: int = 3) -> str:
    """删掉 OCR 认成英文的"伪中文行"。

    扫描页的整页汉字占比通常还在 0.5 以上，页级门槛（`_MIN_CJK_RATIO`）抓不到
    局部乱码；但乱码是**按行**出现的：

        `C1) RA 效力; re`            ← 应为「（五）申请认定财产无主」
        `i. Me. PPR. FAM. WR. Bia. RA.`

    判据是**两个条件同时成立**：汉字占比低 **且** 有一定数量的拉丁字母。
    只看"汉字占比低"会误杀数字型内容 —— 实测 `Q3 营收 1284300 元` 的汉字占比
    只有 0.25，但它是完全正常的文本，被删掉就是丢数据。乱码的真正特征是
    "中文被认成了拉丁字母"，所以必须要求 `min_latin` 个拉丁字符在场。

    短行不动：`1.5%交纳;`、`（三）` 这类要么长度不够 8，要么没有拉丁字母。
    含条号的行也不动，宁可留噪声，不能把「第X条」丢掉（条号丢了这块就永远
    对不上原文）。
    """
    out: List[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if (len(s) >= min_len
                and cjk_ratio(s) < min_ratio
                and len(re.findall(r"[A-Za-z]", s)) >= min_latin
                and not _ARTICLE_HEAD_RE.match(s)):
            continue
        out.append(ln)
    return "\n".join(out)


# 条号行。「第一百二十条之一」这种修订增补也要认。
_ARTICLE_HEAD_RE = re.compile(
    r"^(第[〇零一二三四五六七八九十百千两\d]{1,12}条(?:之[〇零一二三四五六七八九十]{1,3})?)"
    r"[\s　]*(.*)$"
)


def isolate_article_lines(text: str) -> str:
    """把行首的「第X条」拆成独立段落（前后各留一个空行）。

    为什么必须做
    ────────────
    切块器判断"这是哪一条"只看**段首**（`_article_of` 取段落第一行做正则匹配）。
    md 语料里条号和正文天然在同一段（`第三十六条　发明专利申请…`），一切正常。

    但 PDF 抽出来的空行是**不均匀**的：有的地方是

        第七条
        <空行>
        对发明人或者设计人的非职务发明创造专利申请…

    有的地方却是

        任何单位或者个人不得压制。
        第八条
        两个以上单位或者个人合作完成的…

    后一种「第八条」夹在上一段正文和下一段正文中间，既不在段首也不在段尾，
    切块后 `article` 元数据全空 —— 实测专利法 PDF 85 块里带条号的 **0 块**，
    于是按条切块、按条去重、精确找条三条逻辑同时失效。

    **只在 PDF 分支调用**（见 `loaders._load_pdf`）。md 那边条号本来就在段首，
    动它会改变既有 0.894 基线的切块形态，没有必要冒这个险。
    """
    out: List[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        m = _ARTICLE_HEAD_RE.match(s)
        if not m:
            out.append(ln)
            continue
        head, rest = m.group(1), m.group(2).strip()
        if out and out[-1].strip():
            out.append("")               # 条号前必须断开
        out.append(head)
        out.append("")                   # 条号后也断开，正文单独成段
        if rest:
            out.append(rest)
    # 收敛多余空行：本函数对已处理过的文本必须幂等（跑两遍结果一样），
    # 否则"清洗 → 清洗"会在条号周围堆出越来越长的空行。
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out))


def clean_pages(pages: Iterable[str]) -> List[str]:
    """多页清洗：先去版式噪声（要跨页统计），再逐页清洗。"""
    pages = list(pages)
    return [clean_text(p) for p in strip_page_furniture(pages)]
