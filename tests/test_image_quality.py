# -*- coding: utf-8 -*-
"""图片 OCR 结果的质量判据。

背景：tesseract 在横排小字上会把中文认成拉丁字母，产出

    `AM am, (exo maa, mE.`        `BAM me IES Al FSM:`

字数和标点都正常，看不出问题，但它一个汉字都没有。这类块进了索引就是噪声，
而且 rerank 会把它当"普通文本"打分，可能挤掉真正的证据。所以按汉字占比兜住。
"""

from ingest.clean import cjk_ratio


def test_cjk_ratio_normal_article():
    """正常法条汉字占比在 0.8 以上。"""
    text = "第四条 商业银行以安全性、流动性、效益性为经营原则，实行自主经营，自担风险。"
    assert cjk_ratio(text) > 0.8


def test_cjk_ratio_zero_for_latin_noise():
    """OCR 认错成拉丁字母的碎片，汉字占比是 0。"""
    assert cjk_ratio("AM am, (exo maa, mE.") == 0.0
    assert cjk_ratio("BAM me IES Al FSM:") == 0.0


def test_cjk_ratio_ignores_whitespace():
    """空白不该拉低比例 —— 扫描件里的换行很多。"""
    assert cjk_ratio("第一条\n\n  本法\n") == 1.0


def test_cjk_ratio_empty():
    """空文本返回 0，不抛异常（空块会在别处被标成 empty）。"""
    assert cjk_ratio("") == 0.0
    assert cjk_ratio("   \n\t ") == 0.0


def test_cjk_ratio_table_page_above_threshold():
    """表格页数字多、汉字少，但仍在 0.3 以上 —— 门槛不能把表格挡掉。"""
    text = ("| 级数 | 全年应纳税所得额 | 税率 |\n"
            "| 1 | 不超过 36000 元 | 3% |\n"
            "| 2 | 36000 至 144000 元 | 10% |")
    assert cjk_ratio(text) >= 0.3
