# -*- coding: utf-8 -*-
"""多格式解析 + 清洗 + 切块的测试。

为什么这些必须测
────────────────
解析层坏掉是**静默**的：文件在库里，就是答不出来。之前 `PyPDFLoader` 对扫描件
返回 0 字、不报错，这个洞一直没被发现，因为没有任何测试去看"加载出来是不是空的"。

CI 只装 pytest，所以：pypdf / PIL 用 importorskip；OCR 用注入的假函数，
不为测一行逻辑去装系统级 tesseract。
"""

from __future__ import annotations

import os

import pytest

from ingest.chunker import chunk_doc, chunk_docs, stats as chunk_stats
from ingest.clean import clean_pages, clean_text, strip_page_furniture, unmangle_lines
from ingest.loaders import Doc, load_dir, load_file, stats as load_stats


# ========== 清洗 ==========

def test_strip_page_furniture_removes_repeated_header_and_footer():
    """页眉/页码在多页重复 → 删掉；只出现一次的正常句子 → 留下。"""
    pages = [
        "内部资料 禁止外传\n川菜起源于四川。\n第 1 页",
        "内部资料 禁止外传\n粤菜讲究本味。\n第 2 页",
        "内部资料 禁止外传\n江浙菜偏甜。\n第 3 页",
    ]
    out = strip_page_furniture(pages)
    joined = "\n".join(out)
    assert "内部资料" not in joined
    assert "川菜起源于四川" in joined
    assert "粤菜讲究本味" in joined


def test_strip_page_furniture_works_on_two_pages():
    """两页也要处理：在两页里都出现在页首的行，就是页眉。

    第一版把阈值设成"至少 3 页"才统计，结果 2 页的 PDF 页眉全留着——
    而项目里真实存在的对照表 PDF 恰好只有 2 页。
    """
    pages = ["内部资料\n第一页正文", "内部资料\n第二页正文"]
    out = strip_page_furniture(pages)
    joined = "\n".join(out)
    assert "内部资料" not in joined
    assert "第一页正文" in out[0]
    assert "第二页正文" in out[1]


def test_strip_page_furniture_normalizes_page_numbers():
    """页码每页都变，逐字比较永远凑不够阈值，必须先做数字归一化。"""
    pages = [
        f"内部资料\n这是第 {i} 页的正文，讲的是川菜为什么麻辣。\n第 {i} 页"
        for i in (1, 2, 3)
    ]
    out = strip_page_furniture(pages)
    joined = "\n".join(out)
    lines = {ln.strip() for p in out for ln in p.splitlines()}
    assert "内部资料" not in lines
    assert "第 2 页" not in lines          # 页脚整行被删
    # 长正文里的数字是内容，不能因为"归一化后一样"被连坐删掉
    assert "这是第 1 页的正文，讲的是川菜为什么麻辣。" in joined
    assert "这是第 3 页的正文，讲的是川菜为什么麻辣。" in joined


def test_strip_page_furniture_single_page_untouched():
    """只有一页时没有"重复"这回事，一个字都不该删。"""
    pages = ["只有一页\n正文"]
    assert strip_page_furniture(pages) == pages


def test_unmangle_joins_chinese_soft_wrap():
    """PDF 硬换行："加里奥由禁魔石制\\n成" 应该接回去。"""
    assert unmangle_lines("加里奥由禁魔石制\n成，能吸收魔法。") == "加里奥由禁魔石制成，能吸收魔法。"


def test_unmangle_keeps_real_paragraph_break():
    """上一行以句号结尾 → 是真段落边界，不能接。"""
    text = "第一段结束。\n第二段开始。"
    assert unmangle_lines(text) == text


def test_unmangle_fixes_english_hyphen():
    assert unmangle_lines("informa-\ntion retrieval") == "information retrieval"


def test_clean_text_collapses_blank_lines():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


# ========== 多格式加载 ==========

def _make_image_only_pdf(path: str) -> None:
    """造一页"扫描件"：只有图像、没有文字层。"""
    Image = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")
    img = Image.new("RGB", (600, 300), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 40), "Sales Report Q3", fill="black")
    img.save(path, "PDF")


def test_scanned_pdf_is_flagged_not_silently_empty(tmp_path):
    """核心回归：扫描件必须被标成 needs_ocr/empty，而不是悄悄产出空文档。

    旧实现（PyPDFLoader）在这里返回 text=''，且没有任何标记 —— 后续切块会
    跳过它，表现就是"库里有文件但永远答不出来"。
    """
    pytest.importorskip("pypdf")
    pytest.importorskip("PIL")
    pdf = tmp_path / "scan.pdf"
    _make_image_only_pdf(str(pdf))

    docs = load_file(str(pdf))
    assert len(docs) == 1
    assert docs[0].text.strip() == ""
    assert docs[0].metadata["empty"] is True
    assert docs[0].metadata["needs_ocr"] is True


def test_scanned_pdf_recovers_text_via_injected_ocr(tmp_path):
    """注入 OCR 后，同一份扫描件应该被救回来。"""
    pytest.importorskip("pypdf")
    pytest.importorskip("PIL")
    pdf = tmp_path / "scan.pdf"
    _make_image_only_pdf(str(pdf))

    docs = load_file(str(pdf), ocr=lambda img: "Q3 营收 1284300 元")
    assert "1284300" in docs[0].text
    assert docs[0].metadata.get("ocr_used") is True
    assert docs[0].metadata["empty"] is False


def test_image_file_uses_caption_first(tmp_path):
    """图片优先用 VLM 图注（懂图表），没有图注才退到 OCR。"""
    pytest.importorskip("PIL")
    from PIL import Image

    p = tmp_path / "chart.png"
    Image.new("RGB", (50, 50), "white").save(p)

    docs = load_file(
        str(p),
        caption=lambda img: "柱状图：Q3 营收同比增长 18.4%",
        ocr=lambda img: "不该走到这里",
    )
    assert "18.4%" in docs[0].text
    assert docs[0].metadata.get("caption_used") is True


def test_image_without_any_extractor_is_flagged(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    p = tmp_path / "chart.png"
    Image.new("RGB", (50, 50), "white").save(p)
    docs = load_file(str(p))
    assert docs[0].metadata["empty"] is True
    assert docs[0].metadata["needs_ocr"] is True


def test_load_dir_mixes_formats_and_skips_unknown(tmp_path):
    pytest.importorskip("pypdf")
    pytest.importorskip("PIL")
    (tmp_path / "a.txt").write_text("川菜起源于四川。", encoding="utf-8")
    (tmp_path / "b.md").write_text("# 粤菜\n讲究本味。", encoding="utf-8")
    _make_image_only_pdf(str(tmp_path / "c.pdf"))
    (tmp_path / "d.bin").write_bytes(b"\x00\x01")

    docs = load_dir(str(tmp_path), on_error=lambda p, e: None)
    kinds = {d.metadata["source_file"] for d in docs}
    assert {"a.txt", "b.md", "c.pdf"} <= kinds
    assert "d.bin" not in kinds

    s = load_stats(docs)
    assert s["pages"] == 3
    assert s["empty"] == 1        # 扫描件被数出来了，而不是消失
    assert s["chars"] > 0


def test_pdf_table_extracted_as_markdown(tmp_path):
    """PDF 里的表格要变成 markdown 表，且正文不被去重逻辑误伤。

    回归点：第一版按"单元格"去重，把正文标题里的「菜系」「代表菜」各删了一次，
    "主要菜系对照表" 变成 "主要对照表"。现在按整行拼接去重，短词不受影响。
    """
    pytest.importorskip("pypdf")
    pdfplumber = pytest.importorskip("pdfplumber")
    pytest.importorskip("reportlab")

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "table.pdf"
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    c = canvas.Canvas(str(pdf), pagesize=A4)
    c.setFont("STSong-Light", 11)
    c.drawString(50, 760, "主要菜系对照表")
    rows = [("菜系", "口味特点", "代表菜"), ("川菜", "麻辣鲜香", "开水白菜")]
    y = 700
    for row in rows:
        x = 50
        for cell in row:
            c.rect(x, y - 18, 110, 20)
            c.drawString(x + 4, y - 12, cell)
            x += 110
        y -= 20
    c.save()

    with pdfplumber.open(str(pdf)) as p:
        assert p.pages[0].extract_tables()          # 解析器本身能认出来

    docs = load_file(str(pdf))
    text = docs[0].text
    assert "| 菜系 | 口味特点 | 代表菜 |" in text   # 结构化表进来了
    assert "主要菜系对照表" in text                   # 正文称谓没被误删
    assert docs[0].metadata.get("tables") == 1


# ========== 切块 ==========

def test_chunk_carries_heading_path():
    doc = Doc("# 川菜\n\n## 代表菜\n\n开水白菜是国宴菜。", {"source_file": "chuancai.md"})
    chunks = chunk_doc(doc)
    assert chunks
    assert any("代表菜" in c.metadata["heading_path"] for c in chunks)
    assert chunks[0].metadata["source_file"] == "chuancai.md"


def test_chunk_keeps_table_atomic():
    """表格不能被切成两半——切碎了后半张表读不懂。"""
    table = "\n".join([
        "| 菜名 | 口味 |",
        "| --- | --- |",
        "| 开水白菜 | 清鲜 |",
        "| 麻婆豆腐 | 麻辣 |",
    ])
    doc = Doc("# 川菜\n\n" + table, {})
    chunks = chunk_doc(doc, size=200)
    tables = [c for c in chunks if c.metadata.get("is_table")]
    assert len(tables) == 1
    assert "开水白菜" in tables[0].text and "麻婆豆腐" in tables[0].text
    assert "菜名" in tables[0].text


def test_oversized_table_split_repeats_header():
    rows = ["| 菜名 | 口味 |", "| --- | --- |"] + [f"| 菜{i} | 口味{i} |" for i in range(40)]
    doc = Doc("\n".join(rows), {})
    chunks = chunk_doc(doc, size=200)
    tables = [c for c in chunks if c.metadata.get("is_table")]
    assert len(tables) > 1
    assert all("菜名" in c.text for c in tables)   # 每块都有表头


def test_chunk_respects_size_budget():
    doc = Doc("\n\n".join(f"第{i}段：" + "内容" * 120 for i in range(3)), {})
    chunks = chunk_doc(doc, size=200, overlap=40)
    assert all(len(c.text) <= 400 for c in chunks)   # 允许重叠带来的少量超出
    assert len(chunks) >= 3


def test_empty_doc_produces_no_chunks():
    """空页不产块（但它的 empty 标记已经在 loaders 里留下来了）。"""
    assert chunk_docs([Doc("", {"empty": True})]) == []


def test_chunk_stats_counts_tables():
    docs = [Doc("# 标题\n\n| a | b |\n| --- | --- |\n| 1 | 2 |", {})]
    s = chunk_stats(chunk_docs(docs))
    assert s["tables"] == 1
    assert s["with_heading"] >= 1
