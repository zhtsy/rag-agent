#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成「企业知识库」合成语料 —— 专门用来把索引压到 3–5k 块做规模实验。

为什么需要它
───────────
现有真实语料只有 22 个业务 txt（54 KB）+ 9 篇项目文档（188 KB），
切出来 145 块。要回答「语料涨到几千块，召回还稳不稳、延迟涨多少」，
必须有几千块；而这台机器只通 PyPI（维基/GitHub/Gitee 全部超时），
下不了公开中文语料。所以这里**合成**一批企业知识库文档。

诚实标注（README 与答辩口径都必须一致）
────────────────────────────────────
- `knowledge_base/`、仓库根 `*.md`：**真实**文本，48 题基线用的就是它们
- `knowledge_base_v3/`：**全部由本脚本生成**，合成文本
  用途是压规模、测召回衰减，**不能**用它宣称业务准确率
- 合成语料的价值在「结构真实」：同一批客户/合同/产品/供应商在多个文档里
  交叉出现，金额、人名、参数互不相同且唯一 —— 这样「检索到错的那份」
  是能被判出来的，而堆同一句话几百遍的假语料做不到。

设计要点
────────
1. **事实唯一**：每个客户有独立对接人、每个合同有独立金额、每个产品有独立
   吞吐量。语料内部若出现重复值，QA 校验会报错（而不是悄悄算对）。
2. **多格式**：md / txt / pdf（含表格）/ docx / 图片 png / 纯图像扫描件 pdf，
   比例接近真实企业库（表格与扫描件是解析链路最容易翻车的地方）。
3. **确定性**：固定 seed，同一份脚本任何时候跑出来字节一致，方便复现。
4. **自带标注**：同时产出 `qa_spec.json`（40 条问题 + 金标文件 + 必含答案），
   规模曲线实验直接读它，不用人工标注。

用法：
    python scripts/gen_corpus_v3.py                 # 生成到 knowledge_base_v3/
    python scripts/gen_corpus_v3.py --scale 2.0     # 文档量翻倍（压更大规模）
    python scripts/gen_corpus_v3.py --check         # 只校验已有语料的事实唯一性
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "knowledge_base_v3")
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

SEED = 20261112


# ---------------------------------------------------------------- 实体池

# 真实城市与省份，两跳问题（城市→省）与现有图谱同构，便于对照
CITY_PROV: List[Tuple[str, str]] = [
    ("南京", "江苏"), ("苏州", "江苏"), ("杭州", "浙江"), ("宁波", "浙江"),
    ("深圳", "广东"), ("广州", "广东"), ("成都", "四川"), ("武汉", "湖北"),
    ("青岛", "山东"), ("西安", "陕西"), ("合肥", "安徽"), ("长沙", "湖南"),
]

INDUSTRY = [
    "装备制造", "连锁零售", "冷链物流", "医药流通", "教育科技",
    "新能源", "汽车零部件", "食品加工",
]

SURNAME = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
GIVEN = [
    "伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "洋", "勇",
    "艳", "杰", "涛", "明", "超", "秀英", "海燕", "建平", "志强", "晓东",
    "嘉怡", "文博", "思远", "子墨", "雨桐", "一鸣", "若彤", "斯宇", "沐辰", "清和",
]

DEPTS = [
    "售前解决方案部", "交付实施部", "售后服务部", "研发中心",
    "供应链管理部", "财务管理部", "法务合规部", "人力资源部",
]

PRODUCT_LINES = [
    ("智能仓储", "WMS"),
    ("边缘计算", "EDG"),
    ("工业视觉", "IVS"),
    ("能耗管理", "EMS"),
    ("数据中台", "DMP"),
]

MATERIALS = [
    "伺服电机", "工业相机", "激光雷达", "边缘网关", "控制主板",
    "线束组件", "钣金机柜", "光电开关", "减速机", "电源模块",
]

FAULT_ITEMS = [
    ("E101", "视觉定位偏移超限", "工业视觉", "售前解决方案部"),
    ("E102", "边缘网关离线超过 5 分钟", "边缘计算", "研发中心"),
    ("E103", "堆垛机限位信号丢失", "智能仓储", "售后服务部"),
    ("E104", "能耗采集电表通讯中断", "能耗管理", "交付实施部"),
    ("E105", "数据同步任务积压", "数据中台", "研发中心"),
    ("E106", "扫码枪重复触发", "智能仓储", "售后服务部"),
    ("E107", "相机曝光参数漂移", "工业视觉", "交付实施部"),
    ("E108", "网关证书过期", "边缘计算", "研发中心"),
]

PHRASE_OPEN = [
    "本文件用于内部交付与运维参考，涉及数值以最新一版为准。",
    "以下内容经业务与法务双方确认，归档于企业知识库。",
    "本文档由项目组维护，如有变更以变更单为准。",
    "下列信息来自现场实施记录，已做脱敏处理。",
    "本说明面向一线工程师，按步骤执行即可完成处置。",
]

PHRASE_NOTE = [
    "若现场条件与本文不符，请以现场勘察结论为准并回写本文档。",
    "变更需走审批流程，未经审批的参数调整不在质保范围内。",
    "涉及客户数据的操作须留存操作日志，保留期不少于六个月。",
    "同一问题重复出现三次以上，须提交根因分析报告。",
    "跨部门协作以工单为准，口头沟通不作为交付依据。",
]


class Names:
    """生成不重复的中文人名（姓 × 名 组合 + 序号兜底）。"""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: set = set()

    def one(self) -> str:
        for _ in range(200):
            name = self.rng.choice(SURNAME) + self.rng.choice(GIVEN)
            if name not in self.used:
                self.used.add(name)
                return name
        # 兜底：姓 + 双字名 + 序号，保证一定不重复
        while True:
            name = self.rng.choice(SURNAME) + self.rng.choice(GIVEN) + str(len(self.used))
            if name not in self.used:
                self.used.add(name)
                return name


def make_customer_names(n: int, rng: random.Random) -> List[str]:
    """企业名：城市 + 字号 + 行业词 + 后缀，保证唯一且念起来像真的。"""
    prefix = ["华", "中", "新", "远", "鼎", "恒", "联", "正", "天", "金", "瑞", "宏"]
    mid = ["泰", "和", "昌", "运", "达", "盛", "通", "源", "科", "安", "信", "捷"]
    tail = ["机械", "商贸", "物流", "医药", "科技", "能源", "食品", "电子", "智能", "实业"]
    suffix = ["有限公司", "股份有限公司", "集团有限公司"]
    out, seen = [], set()
    for i in range(n):
        for _ in range(500):
            name = (
                CITY_PROV[i % len(CITY_PROV)][0]
                + rng.choice(prefix) + rng.choice(mid) + rng.choice(tail)
                + rng.choice(suffix)
            )
            if name not in seen:
                seen.add(name)
                out.append(name)
                break
        else:
            out.append(f"样例企业{i:04d}有限公司")
    return out


# ---------------------------------------------------------------- 输出工具

def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_pdf(path: str, lines: List[str], table: List[List[str]] | None = None,
               font_size: int = 11) -> None:
    """带表格的 PDF：报告类与合同类用它，专测 pdfplumber 的表格抽取。"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    os.makedirs(os.path.dirname(path), exist_ok=True)
    pdfmetrics.registerFont(TTFont("WQY", FONT))
    style = ParagraphStyle("cn", fontName="WQY", fontSize=font_size, leading=font_size * 1.7)

    flow = []
    for ln in lines:
        if not ln.strip():
            flow.append(Spacer(1, 6))
        else:
            flow.append(Paragraph(ln.replace("&", "&amp;").replace("<", "&lt;"), style))

    if table:
        rows = [[Paragraph(str(c), style) for c in row] for row in table]
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.6, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ]))
        flow.append(Spacer(1, 10))
        flow.append(t)

    SimpleDocTemplate(path, pagesize=A4, leftMargin=48, rightMargin=48,
                      topMargin=48, bottomMargin=48).build(flow)


def _write_docx(path: str, title: str, lines: List[str],
                table: List[List[str]] | None = None) -> None:
    from docx import Document
    from docx.shared import Pt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = Document()
    doc.add_heading(title, level=1)
    for ln in lines:
        p = doc.add_paragraph(ln)
        for run in p.runs:
            run.font.name = "SimSun"
            run.font.size = Pt(10.5)

    if table:
        t = doc.add_table(rows=0, cols=len(table[0]))
        t.style = "Table Grid"
        for row in table:
            cells = t.add_row().cells
            for i, c in enumerate(row):
                cells[i].text = str(c)
    doc.save(path)


def _write_png(path: str, title: str, rows: List[Tuple[str, int]], unit: str) -> None:
    """真实柱状图 + 中文标注：测「图片只有像素、没有文字层」这条路。"""
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(os.path.dirname(path), exist_ok=True)
    W, H = 960, 600
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT, 26)
    f_body = ImageFont.truetype(FONT, 18)

    d.text((40, 30), title, fill="black", font=f_title)
    base_y, max_h = 500, 340
    top = max(v for _, v in rows) or 1
    x = 110
    for name, v in rows:
        h = int(max_h * v / top)
        d.rectangle([x, base_y - h, x + 110, base_y], outline="black", width=2)
        d.text((x + 8, base_y - h - 28), f"{v:,}", fill="black", font=f_body)
        d.text((x + 20, base_y + 14), name, fill="black", font=f_body)
        x += 200
    d.text((40, 545), f"单位：{unit}", fill="black", font=f_body)
    img.save(path, "PNG")


def _write_scanned_pdf(path: str, lines: List[str]) -> None:
    """整页只有像素、没有文字层的「扫描件」：不 OCR 就抽出 0 字。"""
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT, 34)
    f_body = ImageFont.truetype(FONT, 26)

    y = 150
    for i, ln in enumerate(lines):
        d.text((110, y), ln, fill="black", font=f_title if i == 0 else f_body)
        y += 70 if i == 0 else 52
    img.save(path, "PDF")


# ---------------------------------------------------------------- 各类文档

def gen_manual(i: int, line: str, code: str, model: str, rng: random.Random,
               owner: str) -> Tuple[str, Dict]:
    """产品手册（md）：参数表 + 故障码引用，是「表格查询」的金标来源。"""
    throughput = 800 + i * 37                     # 件/小时，唯一
    precision = round(0.02 + i * 0.001, 3)         # mm，唯一
    power = 1.2 + (i % 9) * 0.3
    fault = FAULT_ITEMS[i % len(FAULT_ITEMS)]

    text = f"""# {model} 产品手册

产品线：{line}　型号：{model}　版本：V{2 + i % 3}.{i % 10}
产品负责人：{owner}

## 一、产品概述

{model} 属于 {line} 产品线，面向{rng.choice(INDUSTRY)}场景，
承担产线末端的自动识别与调度任务。{rng.choice(PHRASE_OPEN)}

## 二、技术参数

| 参数项 | 指标 | 备注 |
| --- | --- | --- |
| 额定吞吐量 | {throughput} 件/小时 | 单通道实测 |
| 定位精度 | ±{precision} mm | 常温常湿 |
| 整机功耗 | {power:.1f} kW | 满载 |
| 工作温度 | -10℃ ~ 45℃ | 需配空调机柜 |
| 通讯协议 | Modbus TCP / OPC UA | 二选一 |

## 三、部署要求

1. 现场需提供 380V 三相电源，容量不低于 {int(power * 2) + 3} kVA。
2. 机柜前后保留 800 mm 检修通道，地面平整度误差不超过 3 mm/m。
3. 与上位系统联调前，须完成 {rng.choice(PRODUCT_LINES)[0]} 侧的点表核对。

## 四、常见故障

现场若出现「{fault[1]}」，先对照故障码 **{fault[0]}** 的标准处置流程执行，
不要自行拆机。处置责任部门为{fault[3]}。

## 五、维护与质保

- 每 {rng.choice([3, 6, 12])} 个月做一次光路清洁与标定。
- 质保期 24 个月，人为损坏与超范围参数调整不在质保内。
- {rng.choice(PHRASE_NOTE)}

> 文档编号 {code}-MAN-{i:03d}　维护人：{owner}
"""
    return text, {"throughput": throughput, "model": model, "owner": owner}


def gen_customer(idx: int, name: str, rng: random.Random, person: str,
                 products: List[str]) -> Tuple[str, Dict]:
    """客户档案（txt）：城市/行业/对接人/在用产品，是「人名查询」的金标来源。"""
    city, prov = rng.choice(CITY_PROV)
    industry = rng.choice(INDUSTRY)
    year = rng.choice([2019, 2020, 2021, 2022, 2023])
    amount = 60 + idx * 13                       # 万元，唯一
    picked = rng.sample(products, k=rng.choice([1, 2, 2, 3]))

    text = f"""客户档案：{name}

客户编号：C{2026}{idx:04d}
注册地：{city}（{prov}省）
所属行业：{industry}
合作起始：{year} 年
年度采购额：{amount} 万元
客户对接人：{person}
我方客户经理：{rng.choice([n for n in ["李承泽", "周敏行", "陈立群", "吴思远"]])}

一、基本情况
{name} 位于{city}，主营{industry}业务，现有产线 {rng.randint(2, 12)} 条，
2025 年起启动数字化改造，重点在仓储调度与质量追溯两个环节。

二、在用产品
{chr(10).join(f"- {p}" for p in picked)}

三、合作记录
{rng.choice(PHRASE_OPEN)}
{rng.choice([
    "一期项目按期验收，客户对交付质量评价为满意，二期正在方案阶段。",
    "一期上线后产能提升约 18%，客户已提出扩点需求，尚在预算审批。",
    "一期出现过两次停机，均按 SOP 在两小时内恢复，客户未提出索赔。",
])}

四、注意事项
{rng.choice(PHRASE_NOTE)}
对接人变更须同步更新本档案，避免工单派错人。
"""
    return text, {"person": person, "city": city, "prov": prov,
                  "industry": industry, "amount": amount}


def gen_contract(no: int, customer: str, models: List[str],
                 rng: random.Random) -> Tuple[List[str], List[List[str]], Dict]:
    """合同摘要（pdf + 表格）：金额唯一，专测表格抽取。"""
    cid = f"HT-2026-{no:04d}"
    fee = 28.5 + no * 3.7                        # 万元，唯一
    days = rng.choice([30, 45, 60, 90])
    sla = rng.choice(["99.5%", "99.0%", "99.9%"])
    sign = f"2026-{(no % 12) + 1:02d}-{(no % 27) + 1:02d}"
    picked = rng.sample(models, k=2)

    lines = [
        "设备采购与实施服务合同（摘要）",
        "",
        f"合同编号：{cid}",
        f"甲方（采购方）：{customer}",
        "乙方（供应方）：本公司",
        f"签署日期：{sign}",
        "",
        "第一条　标的",
        f"甲方向乙方采购 {picked[0]} 与 {picked[1]} 各一套，含现场实施与一年质保。",
        "",
        "第二条　价款与支付",
        f"合同总金额为 {fee:.1f} 万元（含税）。验收合格后 {days} 日内支付 90%，"
        "余款质保期满后结清。",
        "",
        "第三条　服务水平",
        f"乙方承诺系统年可用率不低于 {sla}，故障响应时间不超过 2 小时。",
        "",
        "第四条　违约责任",
        "逾期交付按合同总额每日 0.3% 计违约金，累计不超过 10%。",
        "",
        "第五条　其他",
        rng.choice(PHRASE_NOTE),
    ]
    table = [
        ["条目", "内容"],
        ["合同编号", cid],
        ["合同金额（万元）", f"{fee:.1f}"],
        ["验收后付款天数", str(days)],
        ["可用率承诺", sla],
        ["质保期", "12 个月"],
    ]
    return lines, table, {"cid": cid, "fee": fee, "days": days, "sla": sla,
                          "customer": customer}


def gen_sop(code: str, phenomenon: str, line: str, dept: str,
            rng: random.Random, index: int) -> Tuple[str, Dict]:
    """售后 SOP（md）：故障码 → 处置步骤 → 责任部门，是「故障码查询」的金标。"""
    limit = 15 + index * 5
    text = f"""# 故障码 {code} 处置作业指导书

适用产品线：{line}　责任部门：{dept}　时效要求：{limit} 分钟内首次响应
现象描述：{phenomenon}

## 一、判定条件

监控系统上报 **{code}** 且持续超过 {rng.choice([30, 60, 120])} 秒，即进入本流程。
单次瞬时告警无需处置，但须在工单系统留痕。

## 二、处置步骤

1. 远程确认设备在线状态，读取最近 10 分钟日志。
2. 检查供电与网络链路，确认交换机端口无 err-disable。
3. 若日志出现 `timeout` 关键字，按第 4 步执行；否则跳至第 5 步。
4. 重启采集服务，观察 3 分钟；仍未恢复则升级至二线。
5. 现场检查接插件与接地，必要时更换备件（备件编号 SP-{index:03d}）。
6. 处置完成后回填工单，附日志片段与现场照片。

## 三、升级路径

一线 {limit} 分钟未闭环 → 二线（{dept}）30 分钟介入 →
仍无法定位则拉起研发值班。{rng.choice(PHRASE_NOTE)}

## 四、常见误判

把「{phenomenon}」当成网络抖动是最常见的误判，实际多为传感器老化。
判定依据是日志中是否伴随 `drift` 字样。

> 文档编号 SOP-{code}　修订：2026-{(index % 9) + 1:02d}
"""
    return text, {"code": code, "dept": dept, "limit": limit}


def gen_supplier(i: int, name: str, material: str, rng: random.Random) -> Tuple[str, List[List[str]], Dict]:
    """供应商评估（docx + 表格）：账期唯一。"""
    pass_rate = round(93 + (i % 60) / 10, 1)
    days = 30 + (i % 7) * 15                     # 30/45/60/75/90/105/120
    ontime = round(88 + (i % 11), 1)
    level = "A" if pass_rate >= 96 and ontime >= 95 else ("B" if pass_rate >= 94 else "C")

    lines = [
        f"供应商：{name}",
        f"供应物料：{material}",
        f"评估周期：2026 年上半年",
        "",
        "一、评估结论",
        f"合格率 {pass_rate}%，交付准时率 {ontime}%，综合评级 {level} 级。"
        f"{rng.choice(PHRASE_OPEN)}",
        "",
        "二、商务条款",
        f"结算账期为 {days} 天，月结。价格随行就市，单次调价幅度不超过 5%。",
        "",
        "三、风险提示",
        rng.choice([
            "该供应商为单一来源，需储备至少一家备选。",
            "产能紧张期交付波动明显，建议提前 45 天锁量。",
            "质量稳定，但账期谈判空间有限。",
        ]),
        "",
        "四、改进要求",
        rng.choice(PHRASE_NOTE),
    ]
    table = [
        ["指标", "数值"],
        ["合格率", f"{pass_rate}%"],
        ["交付准时率", f"{ontime}%"],
        ["结算账期（天）", str(days)],
        ["综合评级", f"{level} 级"],
    ]
    return lines, table, {"days": days, "material": material, "level": level}


def gen_policy(i: int, rng: random.Random) -> Tuple[str, str, str]:
    """内部制度（md）：带一张「项目 × 标准」表，数值唯一。

    单位必须按项给：第一版写死"元"，于是「密码更换周期」印成"60 元"、
    「数据导出审批层级」印成"2 元"——语料一眼假，连检索都会因为字段名与
    单位不匹配而失准。
    """
    topics = [
        ("差旅管理办法", "住宿标准", 380 + i * 20, "元", "一线城市每人每晚"),
        ("费用报销管理办法", "单笔招待上限", 500 + i * 50, "元", "业务招待"),
        ("采购审批管理办法", "部门自主采购上限", 2000 + i * 500, "元", "小额采购"),
        ("固定资产管理办法", "资产盘点周期", 3 + i, "个月", "季度盘点"),
        ("信息安全管理办法", "密码更换周期", 60 + i * 5, "天", "登录口令"),
        ("数据分级管理办法", "数据导出审批层级", 1 + i % 3, "级", "二级数据"),
        ("客户拜访管理办法", "拜访记录归档时限", 2 + i, "天", "自然日"),
        ("加班与调休办法", "调休有效期", 60 + i * 10, "天", "自然日"),
        ("培训管理办法", "外部培训报销比例", 50 + i * 5, "%", "学费"),
        ("用车管理办法", "市内用车预约时限", 2 + i, "小时", "工作日"),
    ]
    title, term, value, unit, note = topics[i % len(topics)]
    text = f"""# {title}

适用范围：全体正式员工与外派人员　版本：V{1 + i % 4}.0　生效：2026-0{(i % 9) + 1}-01

## 一、总则

为规范{title[:2]}相关事项，{rng.choice(PHRASE_OPEN)}本制度由财务与人力的共同解释。

## 二、标准

| 项目 | 标准 | 说明 |
| --- | --- | --- |
| {term} | {value} {unit} | {note} |
| 超标处理 | 需事前审批 | 事后补批不予受理 |
| 审批时限 | 3 个工作日 | 超时视为默认通过 |

## 三、流程

1. 员工在系统内提交申请，附必要凭证。
2. 直属主管审批，涉及金额超过 {value} {unit} 须再加一级。
3. 财务复核后于次月 15 日前打款。
4. 争议由{"财务" if i % 2 else "人力"}裁定。

## 四、其他

{rng.choice(PHRASE_NOTE)}
"""
    return text, term, f"{value} {unit}"


def gen_weekly(i: int, project: str, owner: str, rng: random.Random) -> Tuple[str, str]:
    """项目周报（txt）：风险描述唯一，专测「非结构化长句」检索。"""
    risk = rng.choice([
        "客户侧网络割接窗口未确认，可能推迟上线",
        "关键备件到货晚于计划两周，影响联调",
        "甲方数据接口字段缺失，需二次开发",
        "现场电力容量不足，需协调增容",
        "验收标准未对齐，存在返工风险",
        "第三方扫码设备兼容性问题待验证",
        "项目组成员被抽调，人力缺口约 1.5 人月",
    ])
    week = 20 + (i % 20)
    text = f"""项目周报：{project}

项目编号：PJ-2026-{i:04d}
项目经理：{owner}
报告周期：2026 年第 {week} 周

一、本周进展
1. 完成设备到场清点，到货率 {70 + i % 30}%。
2. 完成基础网络布线，机柜上电自检通过。
3. 与客户 IT 完成两轮接口联调，主流程已跑通。

二、风险与问题
本周主要风险是：{risk}。已同步客户项目经理，约定下周例会专题讨论。

三、下周计划
1. 完成剩余点位的安装与标定。
2. 输出联调报告与问题清单。
3. 启动操作人员培训。

四、需要支持
{rng.choice(PHRASE_NOTE)}
"""
    return text, risk


def gen_report(quarter: int, region: str, rng: random.Random) -> Tuple[List[str], List[List[str]], Dict]:
    """季度经营报告（pdf + 表格）：区域营收唯一，专测「数值型查询」。"""
    base = {"华东": 4280, "华南": 3160, "华北": 2740, "西南": 1980, "华中": 2210}[region]
    revenue = base + quarter * 130
    yoy = round(8 + (quarter % 4) * 3.4 + rng.random(), 1)
    lines = [
        f"2026 年第{quarter}季度经营简报（{region}区）",
        "",
        f"本季度{region}区实现营业收入 {revenue} 万元，同比增长 {yoy}%。",
        "增长主要来自存量客户的扩点需求，新签客户贡献有限。",
        "",
        "一、收入结构",
        "硬件销售占比 46%，实施服务占比 31%，年度运维占比 23%。",
        "运维收入占比连续三个季度提升，毛利结构改善。",
        "",
        "二、重点项目",
        f"本季度完成 3 个验收项目，2 个项目处于联调阶段。",
        f"回款率 {72 + quarter * 3}%，较上季度提升 4 个百分点。",
        "",
        "三、下季度目标",
        f"目标收入 {revenue + 180} 万元，重点推进存量客户二期。",
        "",
        rng.choice(PHRASE_NOTE),
    ]
    table = [
        ["指标", "本季度", "上季度", "同比"],
        ["营业收入（万元）", str(revenue), str(revenue - 210), f"{yoy}%"],
        ["新签合同数", str(6 + quarter), str(5 + quarter), "+1"],
        ["回款率", f"{72 + quarter * 3}%", f"{68 + quarter * 3}%", "+4pt"],
    ]
    return lines, table, {"revenue": revenue, "region": region, "quarter": quarter}


# ---------------------------------------------------------------- 主流程

def build(scale: float = 1.0) -> Dict:
    rng = random.Random(SEED)
    names = Names(rng)

    n_cust = int(240 * scale)
    n_contract = int(200 * scale)
    n_weekly = int(240 * scale)
    n_manual, n_supplier = 60, 60

    customers = make_customer_names(n_cust, rng)

    # 产品：5 条产品线 × 12 型号 = 60
    products, models = [], []
    for line, code in PRODUCT_LINES:
        for k in range(12):
            model = f"{code}-{2000 + k * 100 + (k % 3) * 10}"
            products.append((line, model))
            models.append(model)

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = {"manual": 0, "customer": 0, "contract": 0, "sop": 0,
             "supplier": 0, "policy": 0, "weekly": 0, "report": 0,
             "scan": 0, "chart": 0}
    facts: Dict[str, List] = {
        "persons": [], "fees": [], "throughput": [], "fault": [],
        "policy": [], "supplier_days": [], "revenue": [], "risks": [],
    }

    # ---- 产品手册（md）
    manual_meta = []
    for i, (line, model) in enumerate(products):
        owner = names.one()
        text, meta = gen_manual(i, line, model.split("-")[0], model, rng, owner)
        _write_text(os.path.join(OUT_DIR, "manuals", f"{model}.md"), text)
        manual_meta.append((model, line, meta))
        facts["throughput"].append((meta["throughput"], f"{model}.md"))
        stats["manual"] += 1

    # ---- 客户档案（txt）
    cust_meta = []
    for i, name in enumerate(customers, start=1):
        person = names.one()
        text, meta = gen_customer(i, name, rng, person, models)
        fn = f"customer_{i:04d}.txt"
        _write_text(os.path.join(OUT_DIR, "customers", fn), text)
        cust_meta.append((name, fn, meta))
        facts["persons"].append((person, fn))
        stats["customer"] += 1

    # ---- 合同摘要（pdf）
    contract_meta = []
    for i in range(1, n_contract + 1):
        cust = rng.choice(customers)
        lines, table, meta = gen_contract(i, cust, models, rng)
        fn = f"{meta['cid']}.pdf"
        _write_pdf(os.path.join(OUT_DIR, "contracts", fn), lines, table)
        contract_meta.append((fn, meta))
        facts["fees"].append((f"{meta['fee']:.1f}", fn))
        stats["contract"] += 1

    # ---- 故障 SOP（md）
    sop_meta = []
    for i, (code, phen, line, dept) in enumerate(FAULT_ITEMS):
        text, meta = gen_sop(code, phen, line, dept, rng, i)
        fn = f"SOP_{code}.md"
        _write_text(os.path.join(OUT_DIR, "sop", fn), text)
        sop_meta.append((code, fn, meta))
        facts["fault"].append((code, fn))
        stats["sop"] += 1

    # ---- 供应商评估（docx）
    supplier_names = make_customer_names(n_supplier, rng)
    supplier_meta = []
    for i, sname in enumerate(supplier_names):
        material = MATERIALS[i % len(MATERIALS)]
        lines, table, meta = gen_supplier(i, sname, material, rng)
        fn = f"supplier_{i:03d}.docx"
        _write_docx(os.path.join(OUT_DIR, "suppliers", fn), sname, lines, table)
        supplier_meta.append((sname, fn, meta))
        facts["supplier_days"].append((meta["days"], fn))
        stats["supplier"] += 1

    # ---- 内部制度（md）
    policy_meta = []
    for i in range(10):
        text, term, value = gen_policy(i, rng)
        fn = f"policy_{i:02d}.md"
        _write_text(os.path.join(OUT_DIR, "policy", fn), text)
        policy_meta.append((term, value, fn))
        facts["policy"].append((term, value, fn))
        stats["policy"] += 1

    # ---- 项目周报（txt）
    weekly_meta = []
    for i in range(1, n_weekly + 1):
        project = f"{rng.choice(CITY_PROV)[0]}{rng.choice(['智能仓储', '能耗监测', '数据中台'])}改造项目"
        project = f"{project}（{i:03d}）"
        owner = names.one()
        text, risk = gen_weekly(i, project, owner, rng)
        fn = f"weekly_{i:04d}.txt"
        _write_text(os.path.join(OUT_DIR, "weekly", fn), text)
        weekly_meta.append((project, fn))
        stats["weekly"] += 1

    # ---- 季度经营报告（pdf）
    report_meta = []
    for quarter in (1, 2, 3):
        for region in ("华东", "华南", "华北", "西南", "华中"):
            lines, table, meta = gen_report(quarter, region, rng)
            fn = f"report_2026Q{quarter}_{region}.pdf"
            _write_pdf(os.path.join(OUT_DIR, "reports", fn), lines, table)
            report_meta.append((fn, meta))
            facts["revenue"].append((meta["revenue"], fn))
            stats["report"] += 1

    # ---- 扫描件（纯图像 pdf）与图表（png）
    for i in range(4):
        lines = [
            "关于设备巡检安排的通知",
            f"各项目组：定于 2026 年 {10 + i} 月开展季度巡检，覆盖 {12 + i * 3} 个在运项目。",
            "巡检重点为光电开关老化、网关证书有效期与备件库存。",
            f"请于 9 月 {20 + i} 日前将巡检计划报至售后服务部。",
        ]
        _write_scanned_pdf(os.path.join(OUT_DIR, "scans", f"notice_scan_{i}.pdf"), lines)
        stats["scan"] += 1

    chart_rows = [("Q1", 4280), ("Q2", 4410), ("Q3", 4580)]
    _write_png(os.path.join(OUT_DIR, "charts", "revenue_trend.png"),
               "2026 年华东区季度营收（万元）", chart_rows, "万元")
    stats["chart"] += 1

    # ---- QA 标注：40 条，金标文件 + 必含答案
    qa = gen_qa(rng, cust_meta, contract_meta, manual_meta,
                sop_meta, supplier_meta, policy_meta, report_meta)

    spec = {
        "seed": SEED,
        "scale": scale,
        "generated_from": "scripts/gen_corpus_v3.py（合成语料，非真实业务数据）",
        "stats": stats,
        "n_files": sum(stats.values()),
        "qa": qa,
    }
    with open(os.path.join(OUT_DIR, "_qa_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)

    return spec


def gen_qa(rng, cust_meta, contract_meta, manual_meta,
           sop_meta, supplier_meta, policy_meta, report_meta) -> List[Dict]:
    """从生成的事实里挑 40 条问题。

    选材原则：问题里必须出现能唯一定位到文档的关键词（客户名 / 合同号 /
    型号 / 故障码 / 制度名 / 区域季度），否则连人都不一定检得回来，
    测的就不是"规模对召回的影响"而是"问题本身不成立"。
    """
    out: List[Dict] = []

    for name, fn, meta in rng.sample(cust_meta, 10):
        out.append({
            "q": f"{name} 的客户对接人是谁？",
            "gold": fn, "key": name, "must": meta["person"], "type": "客户-人名",
        })
    for fn, meta in rng.sample(contract_meta, 8):
        out.append({
            "q": f"合同 {meta['cid']} 的合同总金额是多少万元？",
            "gold": fn, "key": meta["cid"], "must": f"{meta['fee']:.1f}",
            "type": "合同-金额",
        })
    for model, line, meta in rng.sample(manual_meta, 6):
        out.append({
            "q": f"{model} 的额定吞吐量是多少件每小时？",
            "gold": f"{model}.md", "key": model, "must": str(meta["throughput"]),
            "type": "手册-参数",
        })
    for code, fn, meta in rng.sample(sop_meta, 6):
        out.append({
            "q": f"故障码 {code} 由哪个部门负责处置？",
            "gold": fn, "key": code, "must": meta["dept"], "type": "SOP-部门",
        })
    for sname, fn, meta in rng.sample(supplier_meta, 3):
        out.append({
            "q": f"供应商 {sname} 的结算账期是多少天？",
            "gold": fn, "key": sname, "must": str(meta["days"]), "type": "供应商-账期",
        })
    for term, value, fn in rng.sample(policy_meta, 3):
        out.append({
            "q": f"制度里的「{term}」标准是多少？",
            "gold": fn, "key": term, "must": value.split(" ")[0], "type": "制度-标准",
        })
    for fn, meta in rng.sample(report_meta, 4):
        out.append({
            "q": f"2026 年第{meta['quarter']}季度{meta['region']}区的营业收入是多少万元？",
            "gold": fn, "key": meta["region"], "must": str(meta["revenue"]),
            "type": "报告-营收",
        })
    return out


def check() -> int:
    """校验 QA 标注是否可用。

    判两条，都用**解析链路跑出来的真实文本**（不直接读文件：pdf/docx 读不出来，
    第一次就是这么误报了 17 条）：

    1. `key`（合同号 / 型号 / 故障码 / 供应商名 / 区域季度）必须出现在金标文件里
       —— 否则这条问题连人都定位不到，测的不是检索质量
    2. `must`（答案）必须出现在金标文件里

    刻意**不**校验"答案在别的文件里出现了几次"：短数字（800、30 天）在真实
    语料里必然反复出现（检修通道 800 mm、账期 30 天…），按字面查重只会误报。
    检索靠的是 key 词，不是答案字面。
    """
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from ingest import load_dir

    spec_path = os.path.join(OUT_DIR, "_qa_spec.json")
    if not os.path.exists(spec_path):
        print("没有 _qa_spec.json，先跑一次生成")
        return 1
    spec = json.load(open(spec_path, encoding="utf-8"))

    texts: Dict[str, str] = {}
    for src in ("manuals", "customers", "contracts", "sop",
                "suppliers", "policy", "weekly", "reports"):
        d = os.path.join(OUT_DIR, src)
        if not os.path.isdir(d):
            continue
        for doc in load_dir(d, on_error=lambda p, e: None):
            fn = doc.metadata.get("source_file") or ""
            texts[fn] = texts.get(fn, "") + "\n" + doc.text

    bad, soft = 0, 0
    for item in spec["qa"]:
        gold = texts.get(item["gold"], "")
        if not gold:
            print(f"❌ 金标文件解析不出文本: {item['gold']}")
            bad += 1
            continue
        if item["key"] not in gold:
            print(f"❌ 标识词不在金标里: {item['q']} -> {item['gold']}（key={item['key']}）")
            bad += 1
            continue
        if item["must"] not in gold:
            print(f"❌ 答案不在金标里: {item['q']} -> {item['gold']}")
            bad += 1
            continue
        hits = sum(1 for t in texts.values() if item["must"] in t)
        if hits > 50:
            soft += 1           # 只统计，不判失败

    ok = len(spec["qa"]) - bad
    print(f"QA 校验: {ok}/{len(spec['qa'])} 可用"
          f"（另 {soft} 条答案字面较常见，属正常现象，不影响召回判断）")
    return 0 if bad == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser(description="生成企业知识库合成语料")
    ap.add_argument("--scale", type=float, default=1.0, help="文档量倍数")
    ap.add_argument("--check", action="store_true", help="只校验已有语料")
    args = ap.parse_args()

    if args.check:
        return check()

    spec = build(args.scale)
    print(f"📁 输出: {OUT_DIR}")
    print(f"📄 文件 {spec['n_files']} 个：{spec['stats']}")
    print(f"❓ QA 标注 {len(spec['qa'])} 条")
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
