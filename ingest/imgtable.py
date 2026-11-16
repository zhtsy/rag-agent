# -*- coding: utf-8 -*-
"""从图片里还原**表格结构**（不只认字）。

为什么需要单独一层
──────────────────
`tesseract --psm 4` 把整页当"一段文本"读，表格会被摊成一句话：

    乘用车〔按发动机汽缸容量（排气量）分档〕每辆360元至660元

表头、单位、数值糊在一起，检索"1.6 升排量 车船税"时这块永远匹配不上，模型也
看不出哪一列是税额。要拿到结构，必须先知道**格子在哪儿**，再逐格识别。

做法：不引入新模型，纯几何
──────────────────────────
1. 二值化 → 找"长横线"和"长竖线"。判据是**最长连续墨迹长度**，不是墨量：
   一行正文字数再多，也是一个个短笔画，最长连续段不过几十像素；表格框线是
   一条连续的长线。用墨量会把整段密排正文误判成线（试过，满页假表）。
2. 横线聚成带（相邻 ≤ 4px 视为同一条），再按纵向间距把横线分成若干"表"。
3. 每个表内部按竖线切列，横竖交点构成网格。
4. 逐格裁图 → `--psm 7`（单行）识别。空格子（墨量近 0）不送 OCR，避免幻觉。

已知边界（写清楚，免得被当成 bug）
────────────────────────────────
· **不支持合并单元格**：竖线缺失的格子会被当成一行，整张表会错位。真实公文里
  合并表头很常见，这一步需要版面模型（或人工标注）才能可靠判断，当前不做。
· 不支持倾斜扫描（>1°）；不支持无框线表格（和 PDF 那边同一个局限）。
· OCR 出来的数字仍可能有个别字符错（`(=)` 当成 `(二)`），所以表格块的价值在
  "能检索到这张表、能看出行列关系"，数值题仍以文字层 PDF 为准。
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

# 二值化阈值：扫描件背景一般在 220 以上，字在 120 以下，180 是中间的安全位置。
# 太大会把纸纹当墨，太小会丢浅色细线（很多公报表的框线是浅灰）。
_INK_THRESHOLD = 180
# 一条线至少要占表格宽度的这个比例，才算"框线"而不是"一段加粗文字"。
_LINE_RATIO = 0.25
# 横线聚成表时允许的最大行间距（像素）。超了就是两张表。
_TABLE_GAP = 150.0
# 单张表至少几行几列才值得还原（两行的"表"多半是误判）。
_MIN_ROWS = 3
_MIN_COLS = 2


def _binarize(img, max_width: int = 1600):
    import numpy as np

    g = img.convert("L")
    if g.width > max_width:
        r = max_width / float(g.width)
        g = g.resize((max_width, max(1, int(g.height * r))))
    return np.asarray(g) < _INK_THRESHOLD


def _max_run_rows(mask) -> List[int]:
    """每行最长连续 True 的长度。"""
    import numpy as np

    h = mask.shape[0]
    out = [0] * h
    for i in range(h):
        row = mask[i]
        if not row.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[False, row, False].astype(np.int8)))
        if len(edges) >= 2:
            out[i] = int((edges[1::2] - edges[0::2]).max())
    return out


def _max_run_cols(mask) -> List[int]:
    import numpy as np

    w = mask.shape[1]
    out = [0] * w
    for j in range(w):
        col = mask[:, j]
        if not col.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[False, col, False].astype(np.int8)))
        if len(edges) >= 2:
            out[j] = int((edges[1::2] - edges[0::2]).max())
    return out


def _cluster(vals: Sequence[int], gap: int = 4) -> List[int]:
    """把相邻坐标聚成一条线，返回中心坐标。"""
    if not vals:
        return []
    out, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append(int(sum(cur) / len(cur)))
            cur = [v]
    out.append(int(sum(cur) / len(cur)))
    return out


def _group_tables(ys: Sequence[int]) -> List[List[int]]:
    """按纵向间距把横线分成若干张表。"""
    if not ys:
        return []
    groups, cur = [], [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= _TABLE_GAP:
            cur.append(y)
        else:
            groups.append(cur)
            cur = [y]
    groups.append(cur)
    return groups


def _cell_text(img, box, ocr_lang: str) -> str:
    import pytesseract
    from PIL import Image  # noqa: F401

    x0, y0, x1, y1 = box
    # 内缩 3px 去掉框线本身；缩太多会把贴边的字切掉
    crop = img.crop((x0 + 3, y0 + 3, x1 - 3, y1 - 3))
    if crop.width < 8 or crop.height < 8:
        return ""
    try:
        txt = pytesseract.image_to_string(crop, lang=ocr_lang,
                                          config="--psm 7").strip()
    except Exception:
        return ""
    return " ".join(txt.split())


def _clean_cell(s: str) -> str:
    """单元格里的常见 OCR 噪声：竖线、孤立标点、多余空格。"""
    s = (s or "").replace("|", "／").replace("\n", " ")
    s = " ".join(s.split())
    return s.strip(" -—_·.")


def extract_tables(img, max_tables: int = 2) -> List[str]:
    """返回 markdown 表格列表。识别不出就返回 []（调用方退回纯文本 OCR）。"""
    import numpy as np

    mask = _binarize(img)
    if mask.size == 0:
        return []
    h, w = mask.shape

    rows = _max_run_rows(mask)
    cols = _max_run_cols(mask)
    h_cand = [i for i, v in enumerate(rows) if v >= w * _LINE_RATIO]
    v_cand = [j for j, v in enumerate(cols) if v >= h * _LINE_RATIO]
    if len(h_cand) < _MIN_ROWS or len(v_cand) < _MIN_COLS:
        return []

    ys_all = _cluster(h_cand)
    tables: List[str] = []
    lang = os.getenv("RAG_OCR_LANG", "chi_sim+eng")

    for ys in _group_tables(ys_all):
        if len(ys) < _MIN_ROWS:
            continue
        top, bot = ys[0], ys[-1]
        if bot - top < 30:
            continue
        # 竖线只在这个表的纵向范围内找：一条横跨全页的虚线不该被当成分列线
        band = mask[top:bot + 1, :]
        cband = _max_run_cols(band)
        v_local = [j for j, v in enumerate(cband) if v >= (bot - top) * 0.5]
        xs = _cluster(v_local)
        if len(xs) < _MIN_COLS + 1:
            continue

        src = _Scaled(img, mask.shape)
        grid = []
        for i in range(len(ys) - 1):
            row = []
            for j in range(len(xs) - 1):
                row.append(_clean_cell(_cell_text(
                    src, (xs[j], ys[i], xs[j + 1], ys[i + 1]), lang)))
            grid.append(row)
        grid = [r for r in grid if any(c for c in r)]
        if len(grid) < 2:
            continue
        header, body = grid[0], grid[1:]
        md = ["| " + " | ".join(header) + " |",
              "| " + " | ".join(["---"] * len(header)) + " |"]
        md += ["| " + " | ".join(r) + " |" for r in body]
        tables.append("\n".join(md))
        if len(tables) >= max_tables:
            break
    return tables


def _orig(img, mask_shape):
    """把裁切坐标从"缩放后的 mask 空间"换回原图。

    `_binarize` 为了算得快会把图缩到 1600 宽，坐标是在缩图上量的；裁切必须回到
    原图，否则小格子里的字只有几个像素，OCR 全错。
    """
    return _Scaled(img, mask_shape)


class _Scaled:
    """把"缩图坐标系"映射回原图再 crop 的轻量包装。"""

    def __init__(self, img, mask_shape):
        self.img = img
        self.w, self.h = mask_shape[1], mask_shape[0]
        self.sx = img.width / float(self.w)
        self.sy = img.height / float(self.h)

    def crop(self, box):
        x0, y0, x1, y1 = box
        return self.img.crop((int(x0 * self.sx), int(y0 * self.sy),
                              int(x1 * self.sx), int(y1 * self.sy)))
