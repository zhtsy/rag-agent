# -*- coding: utf-8 -*-
"""表格抽取：竖排单元格、合并单元格、全角数字、表名、切块粒度。

为什么单独一个文件
──────────────────
表格这条链路有五个坑，任何一个没修好，表面上都"有字、有块、能搜到词"，
但内容是错的：

  1. **竖排单元格**：《车船税法》附表第一列是旋转 90° 排的，`extract_text`
     按内容流吐字，得到「乘 容 用 量 车 …」—— 每个字都对，顺序全错。
  2. **合并单元格**：`每辆` 只写在第一行，下面 8 行在 PDF 里是"没画字"。
     不按几何填充，第 2 行以后的税额就没有单位。
  3. **真空白 vs 合并空白**：挂车没有子类、摩托车没有子类，这些格子是
     "画了框但没写字"，**不能**被上一行填充，否则会出现「摩托车 | 轮式专用机械车」
     这种事实错误（第一版盲填就踩了）。
  4. **全角数字与空格基线**：表格里满页「１．０升」，不归一化则 BM25 的
     token 是「１」「０」两个，查「1.0升」永远落空。
  5. **切块粒度**：按正文的 200 字预算切表，17 行的税额表被切成 14 块，
     每块一行，行间对应关系全丢。

前四个是"内容对不对"，第五个是"能不能被检索到"。缺任何一个，评测分数都
看不出来（题目里没有表格题），但真实问答会给出错误的税率。
"""

from ingest.chunker import _blocks, _split_table
from ingest.clean import drop_garbage_lines, normalize_fullwidth_alnum
from ingest.loaders import (
    _chars_to_text,
    _cluster_by_x,
    _vertical_cell_text,
)


def _char(text, x0, top, size=9.7):
    """构造一个 pdfplumber 风格的字符字典（只带用到的字段）。"""
    return {"text": text, "x0": x0, "x1": x0 + size * 0.9,
            "top": top, "bottom": top + size, "size": size}


def test_vertical_cell_is_read_left_column_downwards():
    """竖排单元格：列按 x 升序、列内按 top 升序。

    实测数据（车船税法附表 r1c0，42pt 宽 × 147pt 高）：
      左列 x≈19：乘 用 车 〔 按 发 动 机 汽 缸
      右列 x≈33：容 量 （ 排 气 量 ） 分 档 〕
    拼起来是「乘用车〔按发动机汽缸容量（排气量）分档〕」。
    """
    left = "乘用车〔按发动机汽缸"
    right = "容量（排气量）分档〕"
    chars = []
    for i, ch in enumerate(left):
        chars.append(_char(ch, 19, 248 + i * 14.7))
    for i, ch in enumerate(right):
        chars.append(_char(ch, 33, 248 + i * 14.7))
    # 打乱顺序，模拟内容流乱序
    chars = chars[::2] + chars[1::2]
    box = (10, 248, 52, 395)
    assert _vertical_cell_text(chars, box) == "乘用车〔按发动机汽缸容量（排气量）分档〕"


def test_normal_text_cell_is_not_treated_as_vertical():
    """每字一列的横排文本不能被当成竖排（`税 目` 有两列，但每列只有 1 字）。"""
    chars = [_char("税", 84, 228), _char("目", 115, 228)]
    assert _vertical_cell_text(chars, (80, 222, 130, 248)) is None


def test_wide_short_cell_is_not_treated_as_vertical():
    """宽而扁的格子（正常横排）即使字数多也不能进竖排分支。"""
    chars = [_char(ch, 60 + i * 10, 228) for i, ch in enumerate("１．０升以上至")]
    assert _vertical_cell_text(chars, (52, 222, 198, 244)) is None


def test_chars_to_text_keeps_digit_and_cjk_on_same_line():
    """数字与汉字的基线差 8.5pt，不能被断成两行。

    未修复时「1.2元至12元」被拼成「元至元 1.2 12」，税额直接变成错的。
    """
    chars = [
        _char("1", 232.9, 637.3), _char(".", 236.9, 637.3), _char("2", 243.4, 637.3),
        _char("元", 253.9, 628.8), _char("至", 264.4, 628.8),
        _char("1", 274.8, 637.3), _char("2", 280.1, 637.3),
        _char("元", 290.6, 628.8),
    ]
    assert _chars_to_text(chars) == "1.2元至12元"


def test_chars_to_text_drops_blank_chars():
    """空格字符的 top 比汉字低 8.5pt，参与分行会把「税目」拆成三行。"""
    chars = [
        _char("税", 52.2, 607.8), _char(" ", 62.6, 616.3),
        _char(" ", 73.1, 616.3), _char(" ", 83.6, 616.3),
        _char("目", 94.1, 607.8),
    ]
    assert _chars_to_text(chars) == "税目"


def test_chars_to_text_keeps_word_gap_in_latin():
    """英文/数字之间的真实字距要补回空格，否则 `Total Charge` 会粘成一个词。"""
    a = _char("Total", 10, 100)
    a["x1"] = 50
    b = _char("Charge", 56, 100)
    assert _chars_to_text([a, b]) == "Total Charge"


def test_cluster_by_x_groups_nearby_columns():
    cols = _cluster_by_x([_char("a", 10, 0), _char("b", 11.5, 10), _char("c", 40, 20)])
    assert len(cols) == 2
    assert cols[0][0] == 10


def test_fullwidth_digits_and_dot_are_normalized():
    """`１．０升` → `1.0升`；竖排括号 `︵︶︹︺` 也转成横排。"""
    assert normalize_fullwidth_alnum("１．０升以上") == "1.0升以上"
    assert normalize_fullwidth_alnum("乘用车︹按排气量︺") == "乘用车〔按排气量〕"
    # 两侧不是数字的全角点（间隔号）不动
    assert normalize_fullwidth_alnum("第一章．总则") == "第一章．总则"


def test_fullwidth_percent_is_normalized():
    """税率表里的 `50％` 要变成 `50%`，否则用户输入「50%」永远匹配不上。

    这是 T02 用例实测出来的：挂车那行写的是「按照货车税额的 50％计算」，
    检索段关键词 `50%` 直接落空。
    """
    assert normalize_fullwidth_alnum("按照货车税额的 50％计算") == "按照货车税额的 50%计算"
    # 中文标点不能跟着变成半角（会影响切块与 must_contain）
    assert normalize_fullwidth_alnum("合同、票据。") == "合同、票据。"


def test_garbage_lines_are_dropped():
    """中文被认成拉丁字母的整行要删，带条号的行要保。"""
    text = "\n".join([
        "i. Me. PPR. FAM. WR. Bia. RA.",
        "第十一条 iA, SEA, MAR, BER",
        "(六) 申请破产;",
    ])
    out = drop_garbage_lines(text)
    assert "PPR" not in out
    assert "第十一条" in out          # 条号优先，宁可留噪声
    assert "(六) 申请破产;" in out


def test_short_noise_line_is_kept():
    """短行不删：`1.5%交纳;` 汉字占比只有 0.29，但它是有用内容。"""
    assert drop_garbage_lines("1.5%交纳;") == "1.5%交纳;"


def test_table_caption_binds_to_table_block():
    """表格正上方的短行要并进 table 块，不能独立成段。"""
    text = "附：\n车船税税目税额表\n| 税目 | 税率 |\n| --- | --- |\n| 乘用车 | 60元 |"
    blocks = _blocks(text)
    kinds = [b[0] for b in blocks]
    assert kinds.count("table") == 1
    table = [b for b in blocks if b[0] == "table"][0][2]
    assert table.startswith("车船税税目税额表")
    assert "乘用车" in table


def test_long_paragraph_is_not_taken_as_caption():
    """超过 30 字的段落不能被误当表名吞掉。"""
    long_line = "第一条　为了加强税收征收管理，规范税收征收和缴纳行为，保障国家税收收入。"
    text = long_line + "\n| 税目 | 税率 |\n| --- | --- |\n| 乘用车 | 60元 |"
    blocks = _blocks(text)
    para = [b for b in blocks if b[0] == "para"]
    assert para and long_line in para[0][2]


def test_table_is_not_cut_into_one_row_per_chunk():
    """17 行的税额表不能切成 14 块。表格的语义单元是整张表。"""
    rows = ["| 乘用车〔按发动机汽缸容量（排气量）分档〕 | 1.0升（含）以下的 | 每辆 | 60元至360元 | 核定载客人数9人（含）以下 |"]
    rows += [f"| 乘用车 | {i}.0升以上 | 每辆 | {i}00元至{i}60元 | 核定载客人数9人（含）以下 |"
             for i in range(2, 9)]
    table = "车船税税目税额表\n| 税目 |  | 计税单位 | 年基准税额 | 备 注 |\n| --- | --- | --- | --- | --- |\n" + "\n".join(rows)
    pieces = _split_table(table, size=200)
    assert 1 <= len(pieces) <= 3, f"被切成 {len(pieces)} 块"
    for p in pieces:
        assert p.startswith("车船税税目税额表"), "每块都要带表名"
        assert "| 税目 |" in p, "每块都要带表头"


def test_table_with_merged_cells_keeps_rows_self_contained():
    """合并单元格填充后，每行都要能单独读懂（有单位、有税目）。"""
    # 这里只验证切块不破坏行；填充逻辑在 _pdf_tables 里靠几何判据完成，
    # 用真实 PDF 的集成验证见 scripts/diag_pdf_table.py。
    table = "车船税税目税额表\n| 税目 | 计税单位 | 税额 |\n| --- | --- | --- |\n" + "\n".join(
        f"| 乘用车 | 每辆 | {i}00元 |" for i in range(1, 6))
    pieces = _split_table(table, size=200)
    for p in pieces:
        for line in p.splitlines():
            if line.startswith("| 乘用车"):
                assert "每辆" in line
