# -*- coding: utf-8 -*-
"""来源标签回归测试。

事故：换成通过 ingest 管线建的多格式索引后，46 道题返回的 sources 全是
「未知来源」，评测里"检索召回率"直接掉到 0.202。原因不是检索坏了，是
读取端只认 metadata["source"]，而 ingest 写的是 source_file + source_path。
"""
import pytest

# CI 只装 pytest（不装 langchain/torch），rag_agent 在那种环境里 import 不了。
# 与项目其它测试一致：缺依赖就跳过，不要让整条 CI 变红。
pytest.importorskip("langchain")

import rag_agent as R  # noqa: E402


def test_prefers_source_file():
    assert R._doc_source({"source_file": "专利法.md"}) == "专利法.md"


def test_falls_back_to_legacy_source_field():
    assert R._doc_source({"source": "./knowledge_base/test8.txt"}) == "test8.txt"


def test_absolute_path_keeps_only_basename():
    meta = {"source_path": "/home/zh/py_code/knowledge_base_law/民法典_合同编.md"}
    assert R._doc_source(meta) == "民法典_合同编.md"


def test_source_file_wins_over_source_path():
    meta = {"source_file": "劳动合同法.md",
            "source_path": "/tmp/whatever/old_name.md"}
    assert R._doc_source(meta) == "劳动合同法.md"


def test_empty_metadata_says_unknown_but_not_crash():
    assert R._doc_source({}) == "未知来源"
    assert R._doc_source({"source_file": "  "}) == "未知来源"


# ========== PDF 语料的文件名清洗 ==========
#
# PDF 原件叫「中华人民共和国专利法_20201017_ff808081752b7d43.pdf」。直接把
# source_file 摆到来源栏，用户看到的就是一串对象 id —— 他要的是"哪部法"，
# 不是"哪个文件"。清洗必须**只认这个尾巴**，别把普通文件名改坏。

def test_pdf_hash_suffix_stripped():
    meta = {"source_file": "中华人民共和国专利法_20201017_ff808081752b7d43.pdf"}
    assert R._doc_source(meta) == "专利法.pdf"


def test_pdf_name_without_prefix_kept_readable():
    """已经是「某法_日期_hash.pdf」但没带「中华人民共和国」前缀时，
    只去尾巴，不多此一举地删字。"""
    meta = {"source_file": "合伙企业法_20060827_2c909fdd678bf17901678bf63dce034d.pdf"}
    assert R._doc_source(meta) == "合伙企业法.pdf"


def test_markdown_source_untouched():
    """md 语料没有日期尾巴，一个字都不能动，否则评测的 expected_source 会失配。"""
    assert R._doc_source({"source_file": "劳动合同法.md"}) == "劳动合同法.md"
    assert R._doc_source({"source_file": "test8.txt"}) == "test8.txt"


def test_short_hex_is_not_treated_as_hash():
    """名字里本来就有 `_20240101_ab` 这种片段（不是对象 id）时不能误删。"""
    meta = {"source_file": "会议纪要_20240101_ab.pdf"}
    assert R._doc_source(meta) == "会议纪要_20240101_ab.pdf"
