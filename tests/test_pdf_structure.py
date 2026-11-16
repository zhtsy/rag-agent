# -*- coding: utf-8 -*-
"""PDF 文本结构归一化：条号必须落在段落边界上。

背景（真实语料实测）
────────────────────
同一部《专利法》，md 版和 PDF 版抽出来的形态完全不同：

    md:   第三十六条　发明专利的申请人请求实质审查的时候……
    pdf:  任何单位或者个人不得压制。
          第八条
          两个以上单位或者个人合作完成的……

PDF 的「第八条」夹在上一段和下一段正文中间，既不在段首也不在段尾。而切块器
判断"这是哪一条"只看段首 —— 实测未修复前，专利法 PDF 85 块里带条号的 **0 块**，
修复后 113 块里 102 块带上条号。这个数字差决定了按条切块、按条去重、精确找条
三条逻辑是全部失效还是全部有效。
"""

from ingest.clean import clean_text, isolate_article_lines
from ingest.loaders import Doc, load_file


def test_article_between_paragraphs_is_isolated():
    """典型形态：条号被夹在上一段正文和下一段正文之间，前后都没有空行。"""
    text = (
        "任何单位或者个人不得压制。\n"
        "第八条\n"
        "两个以上单位或者个人合作完成的发明创造。"
    )
    out = isolate_article_lines(text)
    assert "\n\n第八条\n\n" in out
    assert out.startswith("任何单位或者个人不得压制。")


def test_article_with_body_on_same_line_split():
    """条号和正文同一行时也要拆开，否则条号进不了块首。"""
    out = isolate_article_lines("第八条 两个以上单位合作完成的发明创造。")
    assert out.startswith("第八条")
    assert "\n\n" in out.split("第八条", 1)[1]
    assert "两个以上单位合作完成的发明创造。" in out


def test_no_duplicate_blank_lines_when_already_isolated():
    """已经独立的条号不能越处理空行越多（幂等）。"""
    once = isolate_article_lines("上一段正文。\n第八条\n下一段正文。")
    twice = isolate_article_lines(once)
    assert once == twice


def test_ordinary_text_untouched():
    """不含条号的普通文本一个字都不动。"""
    text = "这是合同的第一部分，说明双方的权利义务。\n第二段。"
    assert isolate_article_lines(text) == text


def test_article_variant_with_zhi():
    """「第一百二十条之一」这种修订增补形式同样要认。"""
    out = isolate_article_lines("前一段。\n第一百二十条之一\n后一段。")
    assert "第一百二十条之一" in out
    assert "\n\n第一百二十条之一\n\n" in out


# ========== 端到端：真实 PDF ==========

def _real_pdf():
    """取仓库里真实下载的法律 PDF；没下载就跳过（CI 不拉 130 MB 语料）。"""
    import glob
    import os
    import pytest

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = glob.glob(os.path.join(root, "knowledge_base_pdf", "*专利法*.pdf"))
    if not files:
        pytest.skip("未下载 PDF 语料（先跑 scripts/fetch_law_pdfs.py）")
    return files[0]


def test_real_pdf_yields_articles():
    """端到端回归：真实法律 PDF 切完块必须带条号。

    这条测试存在的意义：修复前它是 **0 个条号**。所有单元测试都能过，
    但整条链路实际是断的 —— 只有拿真实文件跑一遍才看得出来。
    """
    pytest = __import__("pytest")
    pytest.importorskip("pypdf")
    from ingest import chunk_docs

    docs = load_file(_real_pdf())
    assert docs, "PDF 解析出 0 页"
    chunks = chunk_docs(docs)
    arts = [c.metadata.get("article") for c in chunks if c.metadata.get("article")]
    assert len(arts) >= 30, f"带条号的块太少：{len(arts)}/{len(chunks)}"
    assert any(a.startswith("第三十六条") for a in arts)
