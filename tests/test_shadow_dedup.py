# -*- coding: utf-8 -*-
"""影子抄本硬去重（_dedup_shadow_docs）的纯逻辑测试。

事故：D007《刑法》第三百五十七条，md 版 0.9935 排第 1、pdf 版 0.9389 排第 2 ——
同一段正文占了 5 个候选里的 2 个，而 pdf 那块尾巴上还粘着下一节标题。
原来的做法是给 pdf 减 0.05 分，只能排后面，拦不住两份同时进 top-5。

这里钉住三种会**静默误删证据**的情形：
  · 候选里一块 md 都没有（整部法只有扫描件）时不许删
  · 跨法同号（到处都有「第十条」）不许互相吞
  · 短块不许走内容覆盖率判据（gram 太少会偶然高覆盖）
"""
import pytest

pytest.importorskip("langchain")

from langchain_core.documents import Document  # noqa: E402

import rag_agent as R  # noqa: E402

MD = "第三百五十七条 本法所称的毒品，是指鸦片、海洛因、甲基苯丙胺（冰毒）、吗啡、大麻、可卡因以及国家规定管制的其他能够使人形成瘾癖的麻醉药品和精神药品。毒品的数量以查证属实的走私、贩卖、运输、制造、非法持有毒品的数量计算，不以纯度折算。"
PDF = ("第三百五十七条\n本法所称的毒品，是指鸦片、海洛因、甲基苯丙胺（冰毒）、吗啡、大麻、"
       "可卡因以及国家规定管制的其他能够使人形成瘾癖的麻醉药品和精神药品。毒品的数量以查证"
       "属实的走私、贩卖、运输、制造、非法持有毒品的数量计算，不以纯度折算。\n第八节\n"
       "组织、强迫、引诱、容留、介绍卖淫罪第三百五十八条")


def _md(text=MD, file="刑法.md", article="第三百五十七条"):
    return Document(page_content=text, metadata={"source_file": file, "article": article})


def _pdf(text=PDF, file="中华人民共和国刑法_20201226_ff808181796a636a0179822a19640c92.pdf"):
    return Document(page_content=text, metadata={"source_file": file, "shadow": 1})


def test_law_key_normalizes_dated_pdf_name():
    """带日期+哈希的 PDF 名要和 md 短名归一到同一个键。"""
    assert R._law_key(_pdf().metadata["source_file"]) == R._law_key("刑法.md") == "刑法"


def test_article_no_from_metadata_and_from_head():
    assert R._article_no_of(_md()) == "第三百五十七条"
    # pdf 块没有 article 元数据，只能从正文首行抽
    assert R._article_no_of(_pdf()) == "第三百五十七条"
    assert R._article_no_of(Document(page_content="没有条号的一段话", metadata={})) == ""


def test_dedup_drops_pdf_when_md_same_article():
    docs = [_md(), _pdf()]
    kept = R._dedup_shadow_docs(docs)
    assert len(kept) == 1
    assert kept[0].metadata.get("shadow") is None      # 留下的是 md


def test_dedup_keeps_pdf_when_no_md_in_candidates():
    """整部法只有扫描件时，PDF 是唯一证据，一块都不能删。"""
    docs = [_pdf(file="中华人民共和国专利法_20201017_ff808081752b7d430175e4651cbd1547.pdf")]
    assert len(R._dedup_shadow_docs(docs)) == 1


def test_dedup_does_not_cross_laws():
    """别的法同名条（同号不同内容）不能被吞掉。"""
    md = _md(text="第十条 本法所称的经营者，是指从事商品生产、经营或者提供服务的自然人、法人和非法人组织。",
             file="反垄断法.md", article="第十条")
    pdf = _pdf(text="第十条 为了维护国家安全，保卫人民民主专政的政权和中国特色社会主义制度，制定本法。",
               file="中华人民共和国国家安全法_20150701_ff8080814e6b1a2c014e6c0a12345678.pdf")
    assert len(R._dedup_shadow_docs([md, pdf])) == 2


def test_dedup_skips_short_blocks():
    """短块 gram 太少，不做内容判据（否则偶然高覆盖会误删）。"""
    md = _md(text="第一条 为了惩罚犯罪，保护人民。", article="第一条")
    pdf = _pdf(text="第一条 为了惩罚犯罪，保护人民。")
    # 条号判据仍然命中 → 删；这条测的是"条号抽不出、文本又太短"的情形
    pdf_nohead = _pdf(text="为了惩罚犯罪 保护人民", file=pdf.metadata["source_file"])
    kept = R._dedup_shadow_docs([md, pdf_nohead])
    assert len(kept) == 2


def test_switch_off_by_ratio(monkeypatch):
    monkeypatch.setattr(R, "_SHADOW_DEDUP_RATIO", 1.1)
    assert len(R._dedup_shadow_docs([_md(), _pdf()])) == 2
