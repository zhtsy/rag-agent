#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PDF 语料用例的检索段复验。

只跑检索（hybrid），不调生成模型，所以一条用例 1 秒级，可以随便重跑。
判据三条，缺一不可：

1. **关键词命中**：`must_contain` 是否都出现在召回文本里；
2. **来源正确**：召回的 `【来源：...】` 里有没有期望的 PDF；
3. **无字形乱码**：召回文本里 `/G12 /G13` 这类子集字体编号一旦成片出现，
   说明这条命中是"字数够多但一个字不是正文"的假命中——比没命中更危险，
   因为它会骗过相关性门槛，让 confidence 看起来很高。

用法：
    python scripts/check_cases_pdf.py                      # 默认查 chroma_db_law_pdf
    python scripts/check_cases_pdf.py --dir chroma_db_law   # 对比旧库
    python scripts/check_cases_pdf.py --raw P04             # 看某条原始召回
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def flatten(text: str) -> str:
    """去掉空白与标点，避免"关键词被换行切开"造成的假失败。"""
    drop = set(" \n\t\u3000，。、；：？！“”‘’（）《》〈〉【】,.;:?!\"'()[]-—…·")
    return "".join(ch for ch in text or "" if ch not in drop)


def garbage_ratio(text: str) -> float:
    """字形编号占全文字符的比例，>0.02 基本可以判定为乱码页。"""
    if not text:
        return 0.0
    tokens = re.findall(r"/G\d+", text)
    return sum(len(t) for t in tokens) / len(text)


def parse_sources(text: str):
    return re.findall(r"【来源：([^】]+)】", text or "")


def law_stem(name: str) -> str:
    """把来源名归一成"这部法"的名字，用来比较来源对不对。

    只比法名、不比后缀，是因为同库去重的规则就是 **md 优先**：
    同一部法既有 md 又有 pdf 时，库里只留 md。所以问《专利法》第三十六条
    召回 `专利法.md` 是**正确**的，判它错等于把设计当 bug。
    另外 PDF 来源还带着 `中华人民共和国` 前缀和下划线，归一化后一并去掉。
    """
    s = os.path.basename(name or "").strip()
    s = re.sub(r"\.(pdf|md|txt|docx)$", "", s, flags=re.I)
    s = s.replace("中华人民共和国", "")
    s = re.sub(r"[\s_\-·]", "", s)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="chroma_db_law_pdf", help="索引目录（相对工程根）")
    ap.add_argument("--cases", default="test_cases_pdf.json")
    ap.add_argument("--raw", default="", help="打印指定用例的原始召回")
    args = ap.parse_args()

    os.environ["RAG_CHROMA_DIR"] = args.dir
    sys.path.insert(0, ROOT)

    import kb_paths

    resolved, note = kb_paths.resolve(ROOT)
    print(f"索引: {os.path.basename(resolved)}  {note or '(直接命中)'}")
    if os.path.basename(resolved) != args.dir:
        print("!! 索引没切过去，下面结果无效")
        return 2

    import rag_agent
    import law_scope

    cases = json.load(open(os.path.join(ROOT, args.cases), encoding="utf-8"))
    sources = rag_agent._all_source_files()
    print(f"用例: {len(cases)} 条   索引来源: {len(sources)} 个文件\n")

    ok = 0
    for c in cases:
        q = c["question"]
        text = rag_agent._search_knowledge_base_impl(q, q)
        flat = flatten(text)
        srcs = parse_sources(text)
        want = c.get("expected_source") or ""
        want_list = want if isinstance(want, list) else [want]
        got = {law_stem(s) for s in srcs}
        src_hit = any(law_stem(w) in got for w in want_list if w)
        ratio = garbage_ratio(text)
        miss = [kw for kw in c.get("must_contain", []) if flatten(kw) not in flat]

        # 这部法本来就不在库里时，"拒答"才是唯一正确的答案——
        # 拿别的法来答才是事故。所以这里换判据，而不是让它挂红。
        absent = law_scope.missing_laws(q, sources)
        if absent:
            refused = "没有收录" in text or "无法引用" in text
            passed = refused
            flag = "PASS" if passed else "FAIL"
            print(f"[{flag}] {c['id']} ({c.get('category','')}) 该法未收录，期望拒答")
            print(f"       实际: {text[:80]!r}")
            ok += passed
            print()
            continue

        passed = not miss and src_hit and ratio < 0.02
        ok += passed
        flag = "PASS" if passed else "FAIL"
        print(f"[{flag}] {c['id']} ({c.get('category','')}) 召回 {len(text)} 字")
        if miss:
            print(f"       缺关键词: {miss}")
        print(f"       期望来源: {want} | 实际: {srcs[:4]}{' ...' if len(srcs) > 4 else ''}")
        if ratio >= 0.02:
            print(f"       !! 字形乱码占比 {ratio:.1%}，疑似假命中")

        if args.raw and c["id"] == args.raw:
            print("------- 原始召回 -------")
            print(text[:1500])
            print("-----------------------")
        print()

    print(f"结果: {ok}/{len(cases)} 通过")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
