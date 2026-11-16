# -*- coding: utf-8 -*-
"""law_scope 的纯逻辑测试：只验字符串归一与拒答边界，不碰索引。

重点是**不能误拒**：拒错了等于把一个能答的问题变成答不了，
而且用户完全不知道是被拦掉的。
"""
from law_scope import law_stem, mentioned_laws, missing_laws

FILES = [
    "中华人民共和国专利法_20201017_ff808081752b7d43.pdf",
    "专利法.md",
    "商业银行法.pdf",
    "民法典_合同编.md",
    "民法典_总则.md",
]


def test_stem_strips_prefix_ext_and_underscore():
    assert law_stem("中华人民共和国专利法_20201017_ff808081752b7d43.pdf")[:5] == "专利法20"
    assert law_stem("专利法.pdf") == "专利法"
    assert law_stem("民法典_合同编.md") == "民法典合同编"
    assert law_stem("《专利法》") == "专利法"


def test_mentioned_laws_skips_amendment_titles():
    q = "根据《关于修改〈中华人民共和国专利法〉的决定》，《专利法》第三十六条怎么规定"
    got = mentioned_laws(q)
    assert "专利法" in got
    # 「决定」是立法文件，不是一部法，不能拿来判"库里有不有"
    assert all("决定" not in g for g in got)


def test_present_law_is_not_missing():
    assert missing_laws("《专利法》第三十六条是怎么规定的？", FILES) == []
    # 库里是「民法典_合同编」，用户只写《民法典》——必须能对上
    assert missing_laws("《民法典》合同编规定合同可以采用哪些形式？", FILES) == []
    assert missing_laws("《中华人民共和国商业银行法》的经营原则是什么？", FILES) == []


def test_absent_law_is_missing():
    assert missing_laws("《计量法》规定国家采用什么计量单位制？", FILES) == ["计量法"]


def test_no_book_title_means_no_judgement():
    # 「专利法第三十六条」这种不带书名号的写法不拦，交给既有链路
    assert missing_laws("专利法第三十六条怎么规定", FILES) == []
    assert missing_laws("", FILES) == []


def test_partial_hit_does_not_refuse():
    # 问两部法、中一部：答复到的那半仍然有价值，不该整体拒答
    assert missing_laws("比较《专利法》和《计量法》的保护对象", FILES) == []


def test_empty_index_fails_open():
    # 索引读不到（清单为空）时宁可放行，也不能把所有问题拒成"没有收录"
    assert missing_laws("《计量法》怎么规定", []) == []
    assert missing_laws("《计量法》怎么规定", ["", None]) == []
