# -*- coding: utf-8 -*-
"""切块正文必须带章节标题。

事故：换真实法律语料后，「《专利法》第三十六条是怎么规定的」检索回了
商标法、刑法、企业所得税法的同号条文 —— 52 部法律里都有「第三十六条」，
而块正文只有条文本身，法名只躺在 metadata 里，两个检索器都看不见。
"""
from ingest.chunker import _with_heading, chunk_doc
from ingest.loaders import Doc


def test_heading_inlined_into_text():
    doc = Doc("# 中华人民共和国专利法\n\n## 第二章 授予专利权的条件\n\n"
              "第三十六条　发明专利的申请人请求实质审查的时候，应当提交资料。",
              {"source_file": "专利法.md"})
    chunks = chunk_doc(doc)
    assert chunks
    # 法名必须出现在正文里，不能只留在 metadata
    assert "专利法" in chunks[0].text
    assert "第三十六条" in chunks[0].text


def test_no_heading_no_prefix():
    doc = Doc("这是一段没有任何标题的正文，长度足够切出一块。", {"source_file": "x.txt"})
    chunks = chunk_doc(doc)
    assert chunks[0].text == "这是一段没有任何标题的正文，长度足够切出一块。"


def test_heading_helper_is_pure():
    assert _with_heading("正文", "") == "正文"
    assert _with_heading("正文", "  ") == "正文"
    assert _with_heading("正文", "第一章") == "第一章\n正文"


def test_table_chunk_also_carries_heading():
    doc = Doc("# 差旅制度\n\n| 城市 | 上限 |\n| --- | --- |\n| 一线 | 500 |",
              {"source_file": "policy.md"})
    chunks = chunk_doc(doc)
    tables = [c for c in chunks if c.metadata.get("is_table")]
    assert tables
    assert all("差旅制度" in c.text for c in tables)
