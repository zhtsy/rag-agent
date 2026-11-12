#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成多格式语料样本：带表格的双页 PDF、扫描件 PDF、图表 PNG。

为什么自己造样本而不是随便下载
─────────────────────────────
要验证的是**解析链路**，不是"能不能下载数据"。所以样本必须能精确控制
三种真实脏版式，而不是碰运气：
  · 每页重复的页眉 + 页码（考验版式噪声剔除）
  · 正文里的表格（考验"表格不被切碎"）
  · 整页只有图像、没有文字层（考验 OCR 兜底，且**不许静默变空**）

产物写进 knowledge_base_v2/，不进生产语料。重复运行是幂等的。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "knowledge_base_v2")


def make_table_pdf(path: str) -> None:
    """双页 PDF：每页带页眉/页码，第 2 页含一张菜系对照表。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4

    pages = [
        ("川菜与粤菜的差异",
         ["川菜以麻辣著称，讲究百菜百味，代表菜有开水白菜、麻婆豆腐。",
          "粤菜讲究本味，清而不淡，代表菜有白切鸡、老火靓汤。",
          "两地气候不同：四川潮湿，需要辛辣驱湿；广东炎热，偏好清淡。"]),
        ("主要菜系对照表",
         ["下表按口味与代表菜对照，数据来自公开资料整理。"]),
    ]

    for idx, (title, paras) in enumerate(pages, start=1):
        # 页眉：两页都一样（考验跨页统计去噪）
        c.setFont("STSong-Light", 9)
        c.drawString(50, h - 40, "内部资料 严禁外传")
        c.drawRightString(w - 50, h - 40, f"第 {idx} 页")
        # 正文
        c.setFont("STSong-Light", 14)
        c.drawString(50, h - 90, title)
        c.setFont("STSong-Light", 11)
        y = h - 130
        for p in paras:
            c.drawString(50, y, p)
            y -= 26
        # 第 2 页画一张表
        if idx == 2:
            rows = [
                ("菜系", "口味特点", "代表菜"),
                ("川菜", "麻辣鲜香", "开水白菜"),
                ("粤菜", "清鲜本味", "白切鸡"),
                ("江浙菜", "清甜软糯", "西湖醋鱼"),
            ]
            y -= 20
            x0, col_w = 50, [80, 130, 130]
            for r_i, row in enumerate(rows):
                x = x0
                for c_i, cell in enumerate(row):
                    c.rect(x, y - 18, col_w[c_i], 20)
                    c.setFont("STSong-Light", 10)
                    c.drawString(x + 5, y - 12, cell)
                    x += col_w[c_i]
                y -= 20
        c.showPage()
    c.save()


def make_scan_pdf(path: str) -> None:
    """整页只有图像、没有文字层 —— 就是"扫描件"。

    必须用真中文字体（文泉驿）：PIL 默认位图字体画不出汉字，会变成方块，
    那样测出来的"OCR 识别率低"是夹具的锅，不是 OCR 的锅。
    """
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(os.path.dirname(path), exist_ok=True)
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    font = ImageFont.truetype(font_path, 36) if os.path.exists(font_path) else None

    img = Image.new("RGB", (1240, 1754), "white")   # 约 A4 @150dpi
    d = ImageDraw.Draw(img)
    lines = [
        "2026 年第三季度经营简报",
        "营业收入 1284300 元，同比增长 18.4%",
        "毛利率 42.7%，环比提升 1.2 个百分点",
        "活跃用户 23150 人，其中新增 4820 人",
    ]
    y = 180
    for ln in lines:
        d.text((120, y), ln, fill="black", font=font)
        y += 90
    img.save(path, "PDF")


def make_chart_png(path: str) -> None:
    """柱状图 + 数字标注，考验 OCR 能不能读出关键数值。

    标签用中文（真中文图表就该是中文）：OCR 文本要能被中文问题命中，
    否则"哪个月营收最高"这种问法在词法上完全对不上，检索必挂——
    这个坑在第一次跑的时候真的踩到了。
    """
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(os.path.dirname(path), exist_ok=True)
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    f_big = ImageFont.truetype(font_path, 30) if os.path.exists(font_path) else None
    f_mid = ImageFont.truetype(font_path, 24) if os.path.exists(font_path) else None

    img = Image.new("RGB", (900, 560), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), "2026 年三季度各月营业收入（元）", fill="black", font=f_big)

    vals = [("7 月", 402000), ("8 月", 431000), ("9 月", 451300)]
    base_y, max_h = 500, 340
    top = max(v for _, v in vals)
    x = 120
    for name, v in vals:
        h = int(max_h * v / top)
        d.rectangle([x, base_y - h, x + 120, base_y], outline="black")
        d.text((x + 10, base_y - h - 34), str(v), fill="black", font=f_mid)
        d.text((x + 30, base_y + 15), name, fill="black", font=f_mid)
        x += 220
    d.text((40, 525), "合计 1284300 元，9 月最高", fill="black", font=f_mid)
    img.save(path, "PNG")


def make_markdown_table(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = """# 经营指标说明

## 口径

营收按含税口径统计，同比基期为 2025 年第三季度。

## 指标表

| 指标 | 2026Q3 | 同比 |
| --- | --- | --- |
| 营业收入 | 1284300 | +18.4% |
| 毛利率 | 42.7% | +1.2pp |
| 活跃用户 | 23150 | +9.8% |

## 备注

毛利率提升主要来自供应链成本下降，与提价无关。
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    make_table_pdf(os.path.join(OUT, "docs", "caixi_duizhao.pdf"))
    make_scan_pdf(os.path.join(OUT, "scans", "q3_jianbao_scan.pdf"))
    make_chart_png(os.path.join(OUT, "scans", "q3_revenue_chart.png"))
    make_markdown_table(os.path.join(OUT, "docs", "jingying_zhibiao.md"))
    print(f"✅ 样本已生成到 {OUT}")
    for cur, _, files in os.walk(OUT):
        for fn in sorted(files):
            p = os.path.join(cur, fn)
            print(f"   {os.path.relpath(p, OUT)}  {os.path.getsize(p)} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
