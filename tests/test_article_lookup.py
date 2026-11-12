# -*- coding: utf-8 -*-
"""「《某法》第 N 条」引用型问句的解析与精确定位。

事故：换真实法律语料后，「《专利法》第三十六条是怎么规定的？」在 4,703 块里
一个专利法的块都进不了 top-5（top1=0.4599，选出来的是民法典物权编）。
52 部法律的条文措辞高度同构，重排只能按"像不像法条"打分。这类查询本质是
精确匹配：法名 → 文件，条号 → 段落。
"""
import pytest

# CI 只装 pytest，rag_agent 依赖 langchain，缺依赖时按项目惯例跳过。
pytest.importorskip("langchain")

import rag_agent as R  # noqa: E402


@pytest.mark.parametrize("q,law,num", [
    ("《专利法》第三十六条是怎么规定的？", "专利法", "三十六"),
    ("劳动合同法第23条说了什么", "劳动合同法", "23"),
    ("民法典第577条", "民法典", "577"),
    ("《中华人民共和国民法典》第五百七十七条", "中华人民共和国民法典", "五百七十七"),
])
def test_regex_extracts_law_and_article(q, law, num):
    m = R._ARTICLE_REF_RE.search(q)
    assert m, f"没匹配上：{q}"
    assert m.group(1) == law
    assert m.group(2) == num


@pytest.mark.parametrize("q", [
    "《专利法》是怎么规定的？",       # 没有条号
    "竞业限制最长几年？",             # 没有法名
    "哪部法律规定，发明专利的申请人请求实质审查的时候？",
])
def test_regex_ignores_non_citation_queries(q):
    assert R._ARTICLE_REF_RE.search(q) is None


def test_int2cn_roundtrip_for_article_numbers():
    assert R._int2cn(5) == "五"
    assert R._int2cn(36) == "三十六"
    assert R._int2cn(577) == "五百七十七"
    assert R._int2cn(1260) == "一千二百六十"
