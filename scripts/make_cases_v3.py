#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从合成语料的标注生成新评测集 `test_cases_v3.json`（48 题）。

为什么不能沿用旧的 48 题
───────────────────────
旧 48 题的答案全在 `knowledge_base/` 那 22 个 txt 里（"小智喜欢什么"）。
把索引切到 v3 语料（3000+ 块、pdf/docx/表格）后，旧题只考到了新增语料的
零头——**测不出新链路（PDF 表格、docx、扫描件）有没有真的可用**。

组成
────
- 40 题来自 `knowledge_base_v3/_qa_spec.json`：答案由生成器保证唯一，
  覆盖客户档案 / 合同 PDF / 产品手册 / 故障 SOP / 供应商 docx / 制度 / 季报。
- 8 题沿用旧集里最能代表真实业务的那几条（人物档案类），保住对
  `knowledge_base/` 的回归——不能因为换了语料就把原有能力测丢了。

`must_not_contain` 只在对"邻近干扰值"时给（比如问华东区营收，禁止答出华南的数字），
不是见谁都塞——过度约束会把答对的判成错。
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "knowledge_base_v3", "_qa_spec.json")
OUT = os.path.join(ROOT, "test_cases_v3.json")

# 旧集里保留的 8 题：真实业务语料，答案短、易判、且不含图谱多跳
KEEP_REAL_IDS = ["T001", "T003", "T006", "T008", "T009", "T010", "T011", "C001"]

CATEGORY = {
    "客户-人名": "直接检索",
    "合同-金额": "表格检索",
    "手册-参数": "表格检索",
    "SOP-部门": "直接检索",
    "供应商-账期": "表格检索",
    "制度-标准": "表格检索",
    "报告-营收": "表格检索",
}


def main() -> int:
    spec = json.load(open(SPEC, encoding="utf-8"))
    cases = []

    for i, item in enumerate(spec["qa"], 1):
        qtype = item["type"]
        cases.append({
            "id": f"V{i:03d}",
            "question": item["q"],
            "expected_answer": f"{item['must']}（来源 {item['gold']}）",
            "expected_source": item["gold"],
            "category": CATEGORY.get(qtype, "直接检索"),
            "must_contain": [item["must"]],
            "must_not_contain": [],
        })

    # 邻近干扰：同一批报表里其他区域的营收，答错就说明召回串了
    by_region = {}
    for item in spec["qa"]:
        if item["type"] == "报告-营收":
            by_region.setdefault(item["gold"], item["must"])
    for case in cases:
        for gold, val in by_region.items():
            if case["expected_source"] == gold:
                others = [v for g, v in by_region.items() if g != gold]
                if others:
                    case["must_not_contain"] = others[:2]

    # 并入真实业务题
    old = {}
    for name in ("test_cases2.json", "test_cases3.json"):
        for c in json.load(open(os.path.join(ROOT, name), encoding="utf-8")):
            old[c["id"]] = c
    for cid in KEEP_REAL_IDS:
        c = old.get(cid)
        if not c:
            print(f"⚠️  旧集里没有 {cid}，跳过")
            continue
        cases.append(dict(c, id=f"R{cid}"))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)

    cats = {}
    for c in cases:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
    print(f"✅ {OUT}")
    print(f"   共 {len(cases)} 题，分类：{cats}")
    print(f"   唯一来源文件 {len({c['expected_source'] if isinstance(c['expected_source'], str) else tuple(c['expected_source']) for c in cases})} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
