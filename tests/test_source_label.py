# -*- coding: utf-8 -*-
"""来源标签回归测试。

事故：换成通过 ingest 管线建的多格式索引后，46 道题返回的 sources 全是
「未知来源」，评测里"检索召回率"直接掉到 0.202。原因不是检索坏了，是
读取端只认 metadata["source"]，而 ingest 写的是 source_file + source_path。
"""
import rag_agent as R


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
