#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 Laws 仓库整理成一套「企业合规知识库」语料，输出到 knowledge_base_law/。

来源：https://github.com/LawRefBook/Laws（10,819 篇中文法律文本，171.8 MB）
取数通道：codeload.github.com 的 tarball（raw.githubusercontent.com 在这台机器上被封）

三条取舍，都是为了"答案可核验"：
1. 同名多版本只留最新 —— 刑法在仓库里有 16 个版本（1979→2023），全收进来
   等于同一部法存 16 份：检索会串台，评测金标也会指到作废版本上。
2. 已被吸收的旧法不收 —— 合同法/物权法/侵权责任法/婚姻法/继承法/民法总则
   已随民法典施行废止，收进来就是主动制造矛盾答案。白名单按名字匹配，天然挡住。
3. 只取"企业真要遵守的那批"，不取 DLC/部门规章全集（那两堆 5,578 + 2,919 篇、
   合计 5,000 万字 ≈ 31 万块，这台 CPU 机器建不完索引，也评测不动）。

用法：
    python scripts/prepare_law_corpus.py --dry-run   # 只看选了什么
    python scripts/prepare_law_corpus.py             # 真正写 knowledge_base_law/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "/home/zh/data/laws_raw/Laws-master"
OUT = os.path.join(ROOT, "knowledge_base_law")

# 企业合规岗位真正会查的法律。刻意不收：证券法/票据法/信托法/保险法这类
# 只对特定行业成立的，以及 DLC（法律法规汇编）里的历史文本。
WANT = [
    # 基础
    "民法典", "宪法", "刑法",
    # 市场主体
    "公司法", "合伙企业法", "个人独资企业法", "企业破产法",
    # 交易与竞争
    "招标投标法", "政府采购法",
    "专利法", "商标法", "著作权法", "反不正当竞争法", "反垄断法",
    "消费者权益保护法", "产品质量法", "食品安全法", "电子商务法",
    # 劳动用工（企业最高频）
    "劳动法", "劳动合同法", "劳动争议调解仲裁法", "社会保险法",
    "安全生产法", "职业病防治法", "工会法",
    # 数据与网络（近年最常被问）
    "个人信息保护法", "数据安全法", "网络安全法",
    # 财税
    "企业所得税法", "个人所得税法", "税收征收管理法", "会计法", "审计法",
    # 环境
    "环境保护法", "环境影响评价法", "固体废物污染环境防治法", "大气污染防治法",
    # 争议解决
    "民事诉讼法", "刑事诉讼法", "行政诉讼法", "仲裁法",
    # 行政
    "行政处罚法", "行政许可法", "行政强制法", "行政复议法",
]

DATE_RE = re.compile(
    r"^(?:中华人民共和国)?(.+?)(?:\((\d{4})-(\d{2})-(\d{2})\))?\.md$")


def scan(src: str):
    """返回 {法名: [(日期, 路径), ...]}，日期缺省记为最旧。"""
    by_name = {}
    for dirpath, _, names in os.walk(src):
        for n in names:
            if not n.endswith(".md") or n.startswith("_"):
                continue
            m = DATE_RE.match(n)
            if not m:
                continue
            name = m.group(1)
            date = "-".join(x for x in m.group(2, 3, 4) if x) or "0000-00-00"
            by_name.setdefault(name, []).append((date, os.path.join(dirpath, n)))
    return by_name


def pick(by_name: dict, src: str):
    """白名单 → [(输出文件名, 法律名, 源路径)]。民法典按编拆，其余取最新版本。"""
    picked, missing = [], []
    for want in WANT:
        if want == "民法典":
            d = os.path.join(src, "民法典")
            if os.path.isdir(d):
                for n in sorted(os.listdir(d)):
                    if n.endswith(".md") and not n.startswith("_"):
                        picked.append((f"民法典_{n[:-3]}.md", f"民法典·{n[:-3]}",
                                       os.path.join(d, n)))
            else:
                missing.append(want)
            continue
        cands = by_name.get(want)
        if not cands:
            missing.append(want)
            continue
        cands.sort()
        picked.append((f"{want}.md", want, cands[-1][1]))
    return picked, missing


def clean(text: str) -> str:
    """去掉仓库自带的 HTML 标记；行尾空白一并收掉。"""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = [ln.rstrip() for ln in text.splitlines()]
    # 折叠 3 个以上连续空行
    out, blank = [], 0
    for ln in lines:
        if ln.strip():
            blank = 0
            out.append(ln)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    src = args.src
    if not os.path.isdir(src):
        print(f"❌ 源目录不存在：{src}\n   先跑：bash scripts/fetch_laws.sh")
        return 2

    by_name = scan(src)
    picked, missing = pick(by_name, src)
    if not picked:
        print("❌ 一部都没选到，检查白名单与文件名格式")
        return 2

    total = sum(len(open(p, encoding="utf-8", errors="ignore").read()) for _, _, p in picked)
    print(f"选中 {len(picked)} 篇 / {total/1e4:.1f} 万字 → 按 200/40 切约 {total//160} 块")
    if missing:
        print("⚠️ 未找到：" + "、".join(missing))

    # 多版本提示：证明"只留最新"不是随手写的
    multi = {k: v for k, v in by_name.items() if len(v) > 1 and k in WANT}
    if multi:
        print("\n同名多版本（只留最新）：")
        for k, v in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:6]:
            print("  %-14s %2d 个版本  %s ~ %s"
                  % (k, len(v), min(v)[0], max(v)[0]))

    if args.dry_run:
        print("\n（--dry-run，未写入磁盘）")
        return 0

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    index = []
    for fname, law, path in picked:
        text = clean(open(path, encoding="utf-8", errors="ignore").read())
        if len(text) < 200:      # 空壳/模板文件不进库
            print(f"  跳过过短文件：{law}（{len(text)} 字）")
            continue
        with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
            f.write(text)
        index.append({"file": fname, "law": law, "chars": len(text),
                      "src": os.path.relpath(path, SRC)})

    meta = {"source": "https://github.com/LawRefBook/Laws",
            "license": "见仓库 LICENSE（法律法规文本本身不受著作权保护）",
            "n_files": len(index), "chars": sum(x["chars"] for x in index),
            "files": index}
    with open(os.path.join(args.out, "_law_index.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"\n✅ 写入 {args.out}：{len(index)} 篇 / {meta['chars']/1e4:.1f} 万字")
    print("   下一步：python scripts/make_cases_law.py   # 从条文机械生成评测集")
    return 0


if __name__ == "__main__":
    sys.exit(main())
