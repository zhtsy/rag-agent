# -*- coding: utf-8 -*-
"""同一条文「md + pdf」并存时的去重规则。

这一层的 bug 都很隐蔽：多留一块不会报错，只会让 top-5 里出现两条一模一样的
证据，把真正需要的另一条挤出去 —— 表现是"答案少了一半"，而没人会想到是去重。
所以每条规则都钉一个测试。
"""

import pytest

from ingest.dedupe import (
    dedupe_by_article,
    drop_shadowed_laws,
    drop_toc_blocks,
    law_name,
    mark_shadowed_laws,
    supplement_stats,
)
from ingest.loaders import Doc


def _doc(text, **meta):
    base = {"source_file": "x.md", "ext": ".md"}
    base.update(meta)
    return Doc(text, base)


# ========== 法名归一化 ==========

def test_law_name_strips_republic_prefix():
    """md 叫「专利法.md」，PDF 叫「中华人民共和国专利法_20201017_ab12.pdf」，
    不归一化永远对不齐，两套内容会同时进库。"""
    assert law_name("中华人民共和国专利法_20201017_ff808081752b7d43.pdf") == "专利法"
    assert law_name("专利法.md") == "专利法"


def test_law_name_keeps_generic_file_names():
    """没有前缀、没有日期后缀的名字原样保留。"""
    assert law_name("劳动合同法.md") == "劳动合同法"


# ========== 按条号去重 ==========

def test_same_article_keeps_markdown():
    """同一条号，md 和 pdf 都在 → 留 md（已验证的那份）。"""
    chunks = [
        _doc("第三十六条 发明专利的申请人请求实质审查的时候…", source_file="专利法.md",
             ext=".md", article="第三十六条"),
        _doc("第三十六条 发明专利的申请人请求实质审查的时候……", source_file="中华人民共和国专利法_20201017_ab12cd34.pdf",
             ext=".pdf", article="第三十六条"),
    ]
    kept, stats = dedupe_by_article(chunks)
    assert len(kept) == 1
    assert kept[0].metadata["source_file"] == "专利法.md"
    assert stats["dropped"] == 1
    assert stats["dup_groups"] == 1


def test_pdf_only_article_survives():
    """md 里没有的条（PDF 独有）必须保留，否则"md 优先"就变成了"PDF 白下"。"""
    chunks = [
        _doc("第三十六条 …", source_file="专利法.md", ext=".md", article="第三十六条"),
        _doc("第九十九条 本决定自公布之日起施行。",
             source_file="中华人民共和国专利法_20201017_ab12cd34.pdf",
             ext=".pdf", article="第九十九条"),
    ]
    kept, _ = dedupe_by_article(chunks)
    assert len(kept) == 2
    assert any(c.metadata["source_file"].endswith(".pdf") for c in kept)


def test_different_laws_same_article_not_merged():
    """「第三十六条」在 52 部法里都有 —— 条号必须和法名一起做键。"""
    chunks = [
        _doc("第三十六条 …", source_file="专利法.md", ext=".md", article="第三十六条"),
        _doc("第三十六条 …", source_file="刑法.md", ext=".md", article="第三十六条"),
    ]
    kept, _ = dedupe_by_article(chunks)
    assert len(kept) == 2


def test_no_article_falls_back_to_text_fingerprint():
    """没有条号的块（前言/章节导语）不能一律当重复删掉：
    md 的前言和 pdf 的前言并不相同，只有正文确实一样才算重复。"""
    same = "本法是为了保护专利权人的合法权益，鼓励发明创造。"
    chunks = [
        _doc(same, source_file="专利法.md", ext=".md"),
        _doc(same, source_file="中华人民共和国专利法_20201017_ab12cd34.pdf", ext=".pdf"),
        _doc("本法是为了促进科学技术进步和经济社会发展。",
             source_file="中华人民共和国专利法_20201017_ab12cd34.pdf", ext=".pdf"),
    ]
    kept, _ = dedupe_by_article(chunks)
    assert len(kept) == 2


def test_prefer_txt_also_counts():
    """prefer 是后缀集合，.txt 同样优先于 .pdf。"""
    chunks = [
        _doc("第五条 …", source_file="某法.txt", ext=".txt", article="第五条"),
        _doc("第五条 …", source_file="中华人民共和国某法_20200101_ab12cd34.pdf",
             ext=".pdf", article="第五条"),
    ]
    kept, _ = dedupe_by_article(chunks)
    assert kept[0].metadata["ext"] == ".txt"


def test_dedupe_is_deterministic():
    """同一批输入跑两次结果必须一致，否则建库不可复现。"""
    chunks = [
        _doc("第五条 …", source_file="中华人民共和国某法_20200101_ab12cd34.pdf",
             ext=".pdf", article="第五条"),
        _doc("第五条 …", source_file="中华人民共和国某法_20200101_cd56ef78.pdf",
             ext=".pdf", article="第五条"),
    ]
    a, _ = dedupe_by_article(chunks)
    b, _ = dedupe_by_article(list(reversed(chunks)))
    assert [c.metadata["source_file"] for c in a] == [c.metadata["source_file"] for c in b]


def test_long_article_split_in_md_keeps_all_chunks():
    """长条文被切成两段、`article` 都是「第三十六条」时，**两段都要留**。

    第一版每组只取第一名，把 md 自己的后一段当重复删了 —— 实测让
    刑法.md 636→515、公司法.md 305→268，定义式得分 0.75→0.5，
    因为定义往往就写在被删掉的那半段里。"""
    chunks = [
        _doc("第三十六条 前半段：申请人可以请求实质审查……", source_file="专利法.md",
             ext=".md", article="第三十六条"),
        _doc("第三十六条 续：国务院专利行政部门应当将申请予以公布……", source_file="专利法.md",
             ext=".md", article="第三十六条"),
        _doc("第三十六条 发明专利的申请人请求实质审查……", source_file="中华人民共和国专利法_20201017_ab12cd34.pdf",
             ext=".pdf", article="第三十六条"),
    ]
    kept, stats = dedupe_by_article(chunks)
    assert len(kept) == 2
    assert all(c.metadata["ext"] == ".md" for c in kept)
    assert stats["dropped"] == 1


def test_pdf_only_article_keeps_all_chunks():
    """只有 PDF 时更不能合并：它是这条的唯一来源，合并就是删内容。"""
    chunks = [
        _doc("第八条 上半……", source_file="中华人民共和国某法_20200101_ab12cd34.pdf",
             ext=".pdf", article="第八条"),
        _doc("第八条 下半……", source_file="中华人民共和国某法_20200101_ab12cd34.pdf",
             ext=".pdf", article="第八条"),
    ]
    kept, stats = dedupe_by_article(chunks)
    assert len(kept) == 2
    assert stats["dropped"] == 0


# ========== 法级让位 ==========

def test_pdf_of_md_covered_law_is_dropped_entirely():
    """md 已经在库里的法，PDF 版本整部丢掉。

    实测刑法 pdf 935 块、md 636 块，条号对不上导致按条去重等于没做，
    47 题召回从 1.000 掉到 0.861。"""
    chunks = [
        _doc("第三百五十七条 …", source_file="刑法.md", ext=".md", article="第三百五十七条"),
        _doc("第三百五十八条 …", source_file="刑法.md", ext=".md", article="第三百五十八条"),
        _doc("第三百五十七条 本法所称毒品…", source_file="中华人民共和国刑法_20201226_ab12cd34.pdf",
             ext=".pdf", article="第三百五十七条"),
        _doc("第三百五十八条 组织他人卖淫…", source_file="中华人民共和国刑法_20201226_ab12cd34.pdf",
             ext=".pdf", article="第三百五十九条"),   # 条号认错也要丢
    ]
    kept, stats = drop_shadowed_laws(chunks)
    assert all(c.metadata["ext"] == ".md" for c in kept)
    assert len(kept) == 2 and stats["dropped"] == 2


def test_pdf_only_law_survives():
    """md 没有这部法（商业银行法、邮政法…）→ PDF 是唯一来源，必须留。"""
    chunks = [
        _doc("第四条 …", source_file="刑法.md", ext=".md", article="第四条"),
        _doc("第四条 商业银行以安全性…", source_file="中华人民共和国商业银行法_20150829_ab12cd34.pdf",
             ext=".pdf", article="第四条"),
    ]
    kept, stats = drop_shadowed_laws(chunks)
    assert len(kept) == 2 and stats["dropped"] == 0
    assert any(c.metadata["ext"] == ".pdf" for c in kept)


def test_shadowing_uses_normalized_law_name():
    """`中华人民共和国刑法_20201226_xx.pdf` 和 `刑法.md` 是同一部法，必须能对上。"""
    chunks = [
        _doc("第一条 …", source_file="刑法.md", ext=".md", article="第一条"),
        _doc("第一条 …", source_file="中华人民共和国刑法_20201226_ab12cd34.pdf",
             ext=".pdf", article="第一条"),
    ]
    _, stats = drop_shadowed_laws(chunks)
    assert stats["dropped"] == 1


# ========== 目录块 ==========

def test_toc_block_dropped():
    toc = "目录第一章 总则第二章 授予专利权的条件第三章 专利的申请第四章 专利申请的审查和批准"
    kept, dropped = drop_toc_blocks([_doc(toc, source_file="中华人民共和国专利法_20201017_ab12cd34.pdf",
                                         ext=".pdf")])
    assert dropped == 1 and not kept


def test_real_article_mentioning_chapters_kept():
    """正文里提到「第X章」但不是在列目录 → 必须保留。"""
    text = "第八条 国务院专利行政部门负责管理全国的专利工作；本章规定的期限自申请日起计算。"
    kept, dropped = drop_toc_blocks([_doc(text, source_file="专利法.md", ext=".md")])
    assert dropped == 0 and len(kept) == 1


def test_long_toc_like_text_kept():
    """超长文本即使含「目录」「章」也不当目录丢 —— 那可能是真正的制度说明。"""
    text = "目录" + "第一章 总则第二章 申请第三章 审查" * 40
    kept, dropped = drop_toc_blocks([_doc(text, source_file="x.pdf", ext=".pdf")], max_chars=400)
    assert dropped == 0 and len(kept) == 1


# ========== 图片文件名归一到法名 ==========

def test_law_name_strips_page_suffix():
    """图片按页导出成 `专利法_p012.png` 时必须归到「专利法」。

    不剥这个尾巴，图片会被当成另一部叫「专利法_p012」的法：条号去重和 md 让位
    全部绕过，同一部法就在库里出第三份抄本 —— 正是把召回压到 0.861 的那个坑。
    """
    assert law_name("专利法_p012.png") == "专利法"
    assert law_name("中华人民共和国专利法_p12.jpg") == "专利法"
    assert law_name("专利法_page7.png") == "专利法"


def test_image_chunk_yields_to_markdown():
    """图片块与 md 同条号时该被丢掉（图片是同一页的另一种封装）。"""
    chunks = [
        _doc("第三十六条 发明专利的申请人请求实质审查的时候…", source_file="专利法.md",
             ext=".md", article="第三十六条"),
        _doc("第三十六条 发明专利的申请人请求实质审查的时候…", source_file="专利法_p012.png",
             ext=".png", article="第三十六条"),
    ]
    kept, _ = dedupe_by_article(chunks)
    assert [c.metadata["ext"] for c in kept] == [".md"]


def test_image_of_shadowed_law_survives():
    """md 有的法，图片**不**整部让位；PDF 才让位。

    图片是人工挑的少数几页，常承载文本层抽不出的表格/印章；跟着 PDF 一起挡掉，
    就等于关掉唯一能补表格的入口。这里把两种格式放一起，验证只丢 pdf。
    """
    chunks = [
        _doc("第一条 …", source_file="个人所得税法.md", ext=".md", article="第一条"),
        _doc("第一条 …", source_file="中华人民共和国个人所得税法_20180831_ab12cd34.pdf",
             ext=".pdf", article="第一条"),
        _doc("个人所得税税率表（综合所得适用）级数 全年应纳税所得额 税率 …",
             source_file="个人所得税法_p005.png", ext=".png"),
    ]
    kept, stats = drop_shadowed_laws(chunks)
    exts = sorted(c.metadata["ext"] for c in kept)
    assert exts == [".md", ".png"]
    assert stats["dropped"] == 1


# ========== 法级标记（2026-11-13 替代整部丢弃） ==========

def test_mark_keeps_pdf_but_flags_it():
    """md 已有的法：PDF 块**保留**，只打 shadow=1；md 块明确置 0。

    存储和召回是两件事。删掉是不可逆的，标记是可回退的排序偏好 —— 检索侧
    按 shadow 减分（同分时 md 胜），PDF 独有的内容依然能被召回。
    """
    chunks = [
        _doc("第一条 …", source_file="刑法.md", ext=".md", article="第一条"),
        _doc("第一条 …", source_file="中华人民共和国刑法_20201226_ab12cd34.pdf",
             ext=".pdf", article="第一条"),
    ]
    kept, stats = mark_shadowed_laws(chunks)
    assert len(kept) == 2, "PDF 不能丢"
    assert stats["shadowed"] == 1
    assert [c.metadata["shadow"] for c in kept] == [0, 1]


def test_mark_ignores_pdf_of_uncovered_law():
    """md 没有的法：PDF 是唯一来源，shadow 必须是 0（否则会被自己降权）。"""
    chunks = [
        _doc("第四条 …", source_file="刑法.md", ext=".md", article="第四条"),
        _doc("第四条 商业银行以安全性…",
             source_file="中华人民共和国商业银行法_20150829_ab12cd34.pdf",
             ext=".pdf", article="第四条"),
    ]
    kept, stats = mark_shadowed_laws(chunks)
    assert stats["shadowed"] == 0
    assert all(c.metadata["shadow"] == 0 for c in kept)


def test_mark_keeps_table_page_of_shadowed_law():
    """md 有这部法、但 md 里没有的那张附表页：保留 + 标记，不丢。

    这是"整部丢弃"最可惜的一类内容 —— 表格往往只在 PDF 里有。
    """
    chunks = [
        _doc("第一条 …", source_file="个人所得税法.md", ext=".md", article="第一条"),
        _doc("| 级数 | 全年应纳税所得额 | 税率 |\n| --- | --- | --- |\n| 1 | 不超过36000元 | 3% |",
             source_file="中华人民共和国个人所得税法_20180831_ab12cd34.pdf", ext=".pdf"),
    ]
    kept, _ = mark_shadowed_laws(chunks)
    assert len(kept) == 2
    assert kept[1].metadata["shadow"] == 1
    assert "3%" in kept[1].text



# ========== 让位代价体检 ==========

def test_supplement_stats_counts_extra_article():
    """md 没有的条号 → 记成 extra_article（真补充）。"""
    chunks = [
        _doc("第一条 …", source_file="专利法.md", ext=".md", article="第一条"),
        _doc("第一条 …", source_file="中华人民共和国专利法_20201017_ab12cd34.pdf",
             ext=".pdf", article="第一条"),
        _doc("第九十九条 …", source_file="中华人民共和国专利法_20201017_ab12cd34.pdf",
             ext=".pdf", article="第九十九条"),
    ]
    s = supplement_stats(chunks)
    assert s["dup_article"] == 1
    assert s["extra_article"] == 1


def test_supplement_stats_separates_table_from_prose():
    """无条号块要分清「表格」和「散文」：表格是独有信息，散文多是封面题注。"""
    chunks = [
        _doc("第一条 …", source_file="个人所得税法.md", ext=".md", article="第一条"),
        _doc("| 级数 | 全年应纳税所得额 | 税率 |\n| 1 | 不超过36000元 | 3% |\n| 2 | 36000至144000元 | 10% |",
             source_file="中华人民共和国个人所得税法_20181123_ab12cd34.pdf", ext=".pdf"),
        _doc("中华人民共和国个人所得税法 1980年9月10日 第五届全国人民代表大会第三次会议通过",
             source_file="中华人民共和国个人所得税法_20181123_ab12cd34.pdf", ext=".pdf"),
    ]
    s = supplement_stats(chunks)
    assert s["table_like"] == 1
    assert s["prose"] == 1


def test_supplement_stats_ignores_laws_without_markdown():
    """md 没有的法，PDF 本来就会进库，不该算成"让位代价"。"""
    chunks = [
        _doc("第一条 …", source_file="中华人民共和国商业银行法_20150829_ab12cd34.pdf",
             ext=".pdf", article="第一条"),
    ]
    s = supplement_stats(chunks)
    assert s["extra_article"] == 0
    assert s["dup_article"] == 0

