#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从法条机械生成评测集 test_cases_law.json（不需要 LLM，金标 100% 可核验）。

为什么能不用 LLM 生成问题：
    法条本身有稳定编号（第三百九十六条）和唯一表述，问题可以由条文反向构造，
    判分关键词直接从原文截取。合成语料那套必须先编一个答案再验，这里不用编。

四类题各自考什么：
    条文定位 16 条 —— 给定法律+条号，要求答出该条内容。考"能不能精确命中一条"。
    定义式   12 条 —— 「本法所称 X，是指…」反过来问「X 是什么」。考语义理解。
    主题定位  8 条 —— 用条文首句问"哪部法律规定了…"。考场外区分度：
                        相似法条（劳动合同法 vs 劳动法）最容易串台。
    拒答      4 条 —— 问不存在的条号。考"检索不到时敢不敢说没有"。
    真实业务  8 条 —— 沿用企业语料的老题，保证换语料后回归还在。

判分关键词怎么选：
    取条文里**全库文档频率最低**的 8 字片段。频率高说明这句话在很多法律里都有
    （"本法自公布之日起施行"），拿它当关键词，答错也能命中，等于白送分。
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "knowledge_base_law")
OUT = os.path.join(ROOT, "test_cases_law.json")

ART_RE = re.compile(r"^第([〇零一二三四五六七八九十百千两]+)条[\s　]*(.*)$")
DEF_RE = re.compile(r"本法所称(.{2,14}?)，是指(.{10,})")
NUM = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
       "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
PUNCT = re.compile(r"[\s，。、；：？！“”‘’（）《》〈〉,.;:?!\"'()\[\]-]+")


def cn2int(s: str) -> int:
    total = section = num = 0
    for ch in s:
        if ch in NUM:
            num = NUM[ch]
        elif ch == "十":
            section += (num or 1) * 10
            num = 0
        elif ch == "百":
            section += (num or 1) * 100
            num = 0
        elif ch == "千":
            section += (num or 1) * 1000
            num = 0
    return total + section + num


def articles(text: str):
    out, cur = [], None
    for line in text.splitlines():
        s = line.strip()
        m = ART_RE.match(s)
        if m:
            if cur:
                out.append(cur)
            cur = [m.group(1), cn2int(m.group(1)), m.group(2).strip()]
            continue
        if s.startswith("#"):
            if cur:
                out.append(cur)
                cur = None
            continue
        if cur and s:
            cur[2] += ("\n" if cur[2] else "") + s
    if cur:
        out.append(cur)
    return [tuple(x) for x in out if len(x[2]) >= 12]


GRAM = 8
_SEG_RE = re.compile(r"[^，。、；：？！“”‘’（）《》〈〉,.;:?!\"'()\[\]\s]+")


def grams_pos(text: str):
    """返回 [(8 字片段, 在原文中的起点, 是否从句首开始)]，**只在标点之间取**。

    为什么不能先去掉标点再滑窗：那样会切出跨标点的片段，例如
    「假冒专利的除依法」——源文是「假冒专利的，除依法承担民事责任」。
    模型老老实实引用原文时带逗号，`kw in answer` 永远不成立，
    于是"证据找对了、答也答对了"却被判 0 分。实测 5 条条文定位题
    就是这么丢的（来源是 专利法.md，得分 0.00）。

    第三个返回值用来优先挑"从句首开始"的片段：从词中间切出来的
    「外为公众所知的技」虽然也是原文子串、判分能用，但看着不像话，
    而且更容易在别处偶然出现。
    """
    out = []
    for m in _SEG_RE.finditer(text or ""):
        seg, base = m.group(0), m.start()
        for i in range(0, max(0, len(seg) - GRAM + 1), 2):
            out.append((seg[i:i + GRAM], base + i, i == 0))
    return out


def grams(text: str):
    return [g for g, _, _ in grams_pos(text)]


def int2cn(n: int) -> str:
    """整数转中文数字（法律条号写法：一千二百六十）。"""
    d = "零一二三四五六七八九"
    if n < 10:
        return d[n]
    if n < 20:
        return "十" + (d[n % 10] if n % 10 else "")
    out = ""
    for unit, val in (("千", 1000), ("百", 100), ("十", 10), ("", 1)):
        q = n // val
        if q:
            out += d[q] + unit
        elif out and n % val:
            out += "零"
        n %= val
    return out.replace("零零", "零").rstrip("零")


def main() -> int:
    idx = json.load(open(os.path.join(DIR, "_law_index.json"), encoding="utf-8"))
    laws = {}          # law name -> (file, [(cn, num, body)])
    for item in idx["files"]:
        p = os.path.join(DIR, item["file"])
        laws[item["law"]] = (item["file"], articles(open(p, encoding="utf-8").read()))

    # ---- 文档频率：这个 8 字片段在多少**条**里出现过 --------------------
    # 必须按"条"统计，不能按"法律"统计。按法律算时，同一部法里两条相似条文
    # （专利法第22条"现有技术"、第23条"现有设计"都以"申请日以前在国内外"开头）
    # 会被当成 df=1 的罕见片段，于是两条题共用同一个判分关键词 —— 答 B 也能
    # 命中 A，分数虚高。
    print("统计 8-gram 文档频率（按条）…")
    df = Counter()
    for law, (f, arts) in laws.items():
        for _, _, body in arts:
            df.update(set(grams(body)))
    print(f"  去重后 {len(df)} 个片段，只出现在一条里的 {sum(1 for v in df.values() if v == 1)} 个")

    def best_phrase(body: str, need: int = 1, head: int = 40) -> str:
        """挑一个够罕见、且靠近条首的片段当判分关键词。

        优先在条首 head 字内选。全文最低频的片段常落在句子中段
        （"或者提供服务的经"这种半句），既不像该条的核心表述，也更容易
        在别的答案里偶然出现，判分会变脆。
        起点直接用原文位置，不做去标点折算 —— 见 grams_pos 的说明。
        """
        cands = [(df[g], 0 if seg_start else 1, pos, g)
                 for g, pos, seg_start in grams_pos(body)
                 if g in df and df[g] <= need]
        if not cands:
            return ""
        pool = [c for c in cands if c[2] < head] or cands
        pool.sort()
        return pool[0][3]

    def clean_lead(body: str) -> str:
        """条首短句，用作"哪部法律规定了…"的题干。

        表格块（个人所得税税率表）和带冒号/括号的条首一律不要：题干里出现换行
        和"（一）"会让问题读起来不像人话，也让检索混进表格块。
        """
        s = body.split("\n")[0]
        s = re.split(r"[。；]", s)[0]
        s = re.split(r"[，：]", s)[0].strip()
        if any(ch in s for ch in "（()【】|、"):
            return ""
        # "…应当具备下列条件"这类断在半句上的说法，读起来不像问题
        if any(w in s for w in ("下列", "以下", "如下")) or s.endswith("的"):
            return ""
        return s if 12 <= len(s) <= 40 else ""

    rng = random.Random(20261112)
    cases = []

    # ---------- 类型 1：条文定位 ------------------------------------------
    direct, defn, topic, refuse = [], [], [], []
    per_law_direct, per_law_defn, per_law_topic = Counter(), Counter(), Counter()
    for law in sorted(laws):
        f, arts = laws[law]
        # 打散顺序再取：按原顺序取会全部落在第 4、5、6 条（前排条文内容最泛），
        # 20 道题变成同一种难度，也测不出"条文散落在语料各处"时的检索。
        shuffled = list(arts)
        rng.shuffle(shuffled)
        for cn, num, body in shuffled:
            if len(body) < 45 or num <= 3:      # 头几条多为"为了…制定本法"
                continue
            # 定义式：本法所称 X，是指 Y
            m = DEF_RE.search(body)
            if m and per_law_defn[law] < 3:
                term = m.group(1).strip("　 的").strip()
                core = best_phrase(m.group(2)[:60], head=30)
                if term and 2 <= len(term) <= 12 and core:
                    per_law_defn[law] += 1
                    defn.append({
                        "id": "D%03d" % (len(defn) + 1),
                        "question": f"《{law}》里的「{term}」是指什么？",
                        "expected_answer": f"见《{law}》第{cn}条：{body[:110]}",
                        "expected_source": f,
                        "category": "定义式",
                        "must_contain": [core],
                        "must_not_contain": [],
                        "_law": law,
                    })
                continue
            # 每部法先各留几条候选，最后再跨法抽样。
            # 之前写成「全局攒满 120 条就停」，结果 Unicode 排序最靠前的专利法、
            # 个人信息保护法把额度吃光，后面 50 部法一条都没进 —— 抽样看起来
            # 是随机的，实际只覆盖了 2 部法。
            if per_law_direct[law] < 6:
                ph = best_phrase(body, need=1)
                if ph:
                    direct.append({
                        "id": "L%03d" % (len(direct) + 1),
                        "question": f"《{law}》第{cn}条是怎么规定的？",
                        "expected_answer": f"第{cn}条：{body[:110]}",
                        "expected_source": f,
                        "category": "条文定位",
                        "must_contain": [ph],
                        "must_not_contain": [],
                        "_law": law, "_num": num,
                    })
                    per_law_direct[law] += 1
            # 主题定位用条文首句，跟上面的判分片段无关，独立攒
            if per_law_topic[law] < 3:
                lead = clean_lead(body)
                if lead:
                    topic.append({
                        "id": "T%03d" % (len(topic) + 1),
                        "question": f"哪部法律规定，{lead}？",
                        "expected_answer": f"《{law}》第{cn}条：{lead}",
                        "expected_source": f,
                        "category": "主题定位",
                        "must_contain": [law.replace("民法典·", "民法典")],
                        "must_not_contain": [],
                        "_law": law,
                    })
                    per_law_topic[law] += 1

    # 分散取题：同一部法律最多出 2 条条文定位，避免全压在民法典上
    print(f"  [debug] 候选：条文定位 {len(direct)} / 定义式 {len(defn)} / "
          f"主题定位 {len(topic)}")
    print("  [debug] 条文定位来自：",
          Counter(c["_law"] for c in direct).most_common(5))

    def spread(items, n, cap=2):
        out, per = [], Counter()
        for it in items:
            if per[it.get("_law", "")] >= cap:
                continue
            out.append(it)
            per[it["_law"]] += 1
            if len(out) >= n:
                break
        return out

    direct = spread(direct, 16)
    defn = spread(defn, 12, cap=2)
    topic = spread(topic, 8, cap=1)

    # ---------- 类型 4：拒答（条号越界 + 主题不在库内） --------------------
    # 条号由各法实际最大条号推出，不写死：写死的话改了语料就会变成"其实存在"。
    for i, key in enumerate(["民法典·合同编", "劳动合同法", "个人信息保护法", "公司法"], 1):
        if key not in laws:
            continue
        arts = laws[key][1]
        # 民法典各编共用一套连续条号（总则 1-204、合同编 463-988…），单看某一编的
        # 最大值会算出"其实存在"的条号；取全法典最大值才安全。
        if key.startswith("民法典"):
            mx = max(a[1] for k, (_, ar) in laws.items() if k.startswith("民法典")
                     for a in ar)
        else:
            mx = max(a[1] for a in arts)
        num = mx + 37
        law = key.split("·")[0]
        refuse.append({
            "id": f"R{i:02d}",
            "question": f"《{law}》第{int2cn(num)}条对碳排放配额的交易规则是怎么规定的？",
            "expected_answer": f"该法共 {mx} 条，第{int2cn(num)}条不存在；且全部语料都没有"
                               f"碳排放配额交易规则，应拒答",
            "expected_source": "",
            "category": "拒答",
            "must_contain": ["没有", "未", "无法", "不存在", "超出"],
            "must_not_contain": [],
        })

    # ---------- 类型 5：真实企业语料（回归） ------------------------------
    real = []
    for name in ("test_cases2.json", "test_cases3.json"):
        p = os.path.join(ROOT, name)
        if not os.path.exists(p):
            continue
        for c in json.load(open(p, encoding="utf-8")):
            must = [k for k in (c.get("must_contain") or []) if k.strip()]
            if not must or c.get("category") == "拒答":
                continue
            real.append({**c, "_file": name, "_must_n": len(must)})
    # 优先取"多关键词、更严格"的题；id 加 B 前缀，避免和上面的 T/L/D/R 撞号
    real.sort(key=lambda c: (-c["_must_n"], c["id"]))
    picked_real = real[:8]
    for i, c in enumerate(picked_real, 1):
        c["id"] = f"B{i:02d}"

    cases = direct + defn + topic + refuse + [
        {k: v for k, v in c.items() if not k.startswith("_")} for c in picked_real
    ]
    for c in cases:
        c.pop("_law", None)
        c.pop("_num", None)

    # ---------- 自检：判分片段不能撞车 --------------------------------
    # 两条题共用关键词时，答对一条会被判成答对两条，评测分数虚高且查不出来。
    seen, dup = set(), []
    for c in cases:
        if c["category"] == "拒答":
            continue
        for kw in c["must_contain"]:
            if kw in seen:
                dup.append((c["id"], kw))
            seen.add(kw)
    if dup:
        print(f"⚠️ {len(dup)} 处关键词重复，丢弃这些题：{dup[:5]}")
        bad = {i for i, _ in dup}
        cases = [c for c in cases if c["id"] not in bad]

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)

    print(f"\n✅ 生成 {len(cases)} 条 → {OUT}")
    for cat, n in Counter(c["category"] for c in cases).most_common():
        print(f"   {cat:<8} {n} 条")
    print("\n抽样：")
    for c in cases[:3] + cases[16:18] + cases[-3:]:
        print(f"   [{c['category']}] {c['question'][:52]}")
        print(f"      must={c['must_contain']}  src={c['expected_source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
