# -*- coding: utf-8 -*-
"""law_scope 的纯逻辑测试：只验字符串归一与拒答边界，不碰索引。

重点是**不能误拒**：拒错了等于把一个能答的问题变成答不了，
而且用户完全不知道是被拦掉的。
"""
from law_scope import law_stem, mentioned_laws, missing_laws, scoped_sources

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


# ========== scoped_sources：点名了库里有的法 → 收敛召回范围 ==========
#
# 与 missing_laws 的分工：
#   missing_laws   → 「一篇都没有」→ 拒答
#   scoped_sources → 「有，但不许别的法混进来」
# 两者必须是同一套判据，否则会出现"既没拒答、又一块都检索不到"。

SCOPE_FILES = [
    "劳动合同法.md",
    "中华人民共和国劳动合同法_20121228_2c909fdd.pdf",
    "劳动合同法实施条例.md",
    "专利法.md",
    "专利法实施细则.md",
    "民法典_合同编.md",
    "民法典_物权编.md",
    "最高人民法院关于适用《民法典》侵权责任编的解释（一）.md",
    "企业所得税法.md",
    "企业所得税法实施条例.md",
    "外商投资企业和外国企业所得税法实施细则.md",
    "印花税法.md",
]


def test_scoped_sources_matches_both_copy():
    """md 和 pdf 是同一部法的两份，都要进范围（否则 md 优先策略失效）。"""
    got = scoped_sources("《劳动合同法》关于试用期的规定", SCOPE_FILES)
    assert "劳动合同法.md" in got
    assert "中华人民共和国劳动合同法_20121228_2c909fdd.pdf" in got


def test_scoped_sources_includes_implementing_rules():
    """实施条例 / 实施细则 / 分编：用户写的是母法，这些必须一起查。"""
    got = scoped_sources("《劳动合同法》关于试用期的规定", SCOPE_FILES)
    assert "劳动合同法实施条例.md" in got

    got = scoped_sources("《专利法》关于新颖性的规定", SCOPE_FILES)
    assert "专利法实施细则.md" in got

    got = scoped_sources("《民法典》关于合同解除的规定", SCOPE_FILES)
    assert "民法典_合同编.md" in got
    assert "民法典_物权编.md" in got


def test_scoped_sources_excludes_judicial_interpretation():
    """司法解释名字里含"民法典"，但它不是民法典本身 —— 不能靠"包含"命中。

    用包含判据时《民法典》会圈进 16 份司法解释（实测 17 个文件），
    收敛等于没做。前缀判据把"中间命中"排除掉。
    """
    got = scoped_sources("《民法典》关于合同解除的规定", SCOPE_FILES)
    assert not any("最高人民法院" in f for f in got)


def test_scoped_sources_excludes_repealed_law():
    """`外商投资企业和外国企业所得税法实施细则` 以"法"结尾，但不以"企业所得税法"开头。

    那部法 2008 年已废止，被"包含"判据误纳入（实测踩到）。
    """
    got = scoped_sources("《企业所得税法》的小型微利企业税率", SCOPE_FILES)
    assert "企业所得税法.md" in got
    assert "企业所得税法实施条例.md" in got
    assert "外商投资企业和外国企业所得税法实施细则.md" not in got


def test_scoped_sources_empty_when_no_book_title():
    """没写书名号 → 不限定范围（"增值税一般纳税人税率"不该被任何范围约束）。"""
    assert scoped_sources("增值税一般纳税人销售货物适用什么税率", SCOPE_FILES) == []


def test_scoped_sources_empty_when_law_absent():
    """点名的法库里没有 → 返回空，交给 missing_laws 去拒答。

    这里**不能**返回一个空范围的"已限定"信号，否则调用方会把
    "没匹配到"误当成"匹配到 0 部法"，检索范围变成空集。
    """
    assert scoped_sources("《计量法》规定什么单位制", SCOPE_FILES) == []


def test_scoped_sources_strips_china_prefix():
    """用户写全称也要能对上（库里可能是短名）。"""
    assert scoped_sources("《中华人民共和国印花税法》税率", SCOPE_FILES) == ["印花税法.md"]
