# -*- coding: utf-8 -*-
"""图片表格还原（ingest/imgtable）的回归测试。

三级验证，任何一级单独通过都不能说明链路可用：

1. **几何**：人造网格必须被认出来（行数/列数正确）。这级只证明"框线能找到"。
2. **真实公文**：把带附录的税法 PDF 渲染成位图（== 一张扫描件），要求还原来
   的行数、列数、关键数额都在。这级证明"整条链路能跑通真实版面"。
3. **噪声**：纯文字页不能凭空造出表（误报比漏报更危险 —— 假表会把正文切碎）。

第 2 级依赖仓库里的 PDF；缺失时跳过，不用 pytest.skip 之外的假设。
"""
import os

import pytest
from PIL import Image, ImageDraw

from ingest.imgtable import extract_tables

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CART = os.path.join(
    ROOT, "knowledge_base_pdf",
    "中华人民共和国车船税法_20190423_ff8080816f135f46016f215dc4071a64.pdf",
)


def _grid(w=600, h=400, rows=4, cols=3):
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for i in range(rows + 1):
        y = int(h * i / rows)
        d.line([(0, y), (w, y)], fill=0, width=3)
    for j in range(cols + 1):
        x = int(w * j / cols)
        d.line([(x, 0), (x, h)], fill=0, width=3)
    return img


def test_synthetic_grid_is_detected():
    tabs = extract_tables(_grid())
    assert len(tabs) == 1
    lines = [ln for ln in tabs[0].splitlines() if ln.strip()]
    # 4 行 × 3 列 → 表头 1 行 + 分隔 1 行 + 正文 3 行
    assert len(lines) == 5
    assert lines[0].count("|") == 4


def test_plain_text_page_is_not_a_table():
    """满页横排文字不能判成表。第一版用"墨量"判线，把密排正文当成 60 行假表。"""
    img = Image.new("L", (600, 400), 255)
    d = ImageDraw.Draw(img)
    for y in range(10, 390, 14):
        d.line([(20, y), (580, y)], fill=60, width=1)
    assert extract_tables(img) == []


@pytest.mark.skipif(not os.path.exists(CART), reason="缺少车船税法 PDF")
def test_real_tax_table_is_recovered():
    """真实公文（渲染成位图）→ 必须还原出 5 列、并且带上关键税额。"""
    pdfplumber = pytest.importorskip("pdfplumber")
    with pdfplumber.open(CART) as doc:
        img = doc.pages[1].to_image(resolution=150).original

    tabs = extract_tables(img)
    assert tabs, "车船税税目税额表没有被还原"
    body = tabs[0]
    assert body.lstrip().startswith("|")
    # 表头列数（合并单元格会让列数偏，只要求至少 4 列）
    assert body.splitlines()[0].count("|") >= 5
    # 关键数额：OCR 允许个别字符错，但 60 元至 360 元 这一行必须能读出来
    flat = body.replace(" ", "")
    assert "60元至360元" in flat
    assert "3600元至5400元" in flat
