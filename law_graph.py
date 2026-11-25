# -*- coding: utf-8 -*-
"""law_graph.py —— 条文引用图：把「这条被谁引用」变成检索能用的证据。

它解决什么问题
──────────────
向量检索只能找**语义像**的条文，找不到**结构上依赖**的条文。

问「《民法典》第五百一十条的规定在哪些情形下适用」：
库里跟它逐字相像的是合同编总则那一片，而真正援引过它的 26 条散布在
买卖、租赁、承揽、仓储各章 —— 它们跟 510 条的**字面相似度极低**，
rerank 永远捞不出来。可它们恰恰就是"这条用在哪"的答案。

这类"谁引用了谁"的知识是**边**，不是文本。语料里那句「依照本法第五百一十条」
确实存在于引用方的原文里，但它提到了 510 条，不等于 510 条的信息能被检索到 ——
检索是单向的：从问题找文本，找不回"谁指向我"。

图的规模决定了它该怎么用
────────────────────────
22,141 个条文节点、1,219 条同法引用边（CITES）、431 条跨法引用边（CITES_LAW）。
出度 0.08，**只有 5% 的条文有出边**。

所以它**不适合**做在线多跳遍历：95% 的问题连第一跳都走不到，
每个请求却都要付一次 Neo4j 往返。这里改成"装载成内存索引"：
一次把全部边拉进进程（约 1,650 行，本地 100~200ms），
之后每次查询都是字典查找（微秒级）。TTL 300 秒，与 graph_entities 一致。

这样它既能进上下文，又不会让问答多等一次网络往返 —— 这是"图"在这套系统里
唯一能接受的形态：**结构化证据的补充源，不是实时的遍历引擎**。

为什么不复用 graph_retriever
────────────────────────────
那套面向 demo 图谱（人物/城市/宠物），三条模板语句把关系名写死了；
法律图的节点是 (:Article)/(:Law)，边是 [:CITES]/[:CITES_LAW]，
两套 schema 没有任何交集。混在一个函数里只会让两边都难改、都难测。

失效与降级
──────────
Neo4j 不可用 → 返回空索引并进入 30 秒冷却，**绝不抛异常**。
法律图是补充证据，它挂了不能拖累向量检索，更不能让整个问答 500。
调用方拿到空结果时行为与"这句话没有引用关系"完全一致。
"""

from __future__ import annotations

import re
import threading
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

from loguru import logger

import law_scope
from graph_retriever import graph

# 与 graph_entities 保持同一口径：改图里的数据后，最多等这么久才生效
_TTL = 300.0
_FAIL_COOLDOWN = 30.0

# 「《专利法》第三十六条」「劳动合同法第23条」这类引用型问句。
# 后缀里必须带「典」：民法典不以「法」结尾，只写 (法|条例|…) 时
# 「民法典第577条」整句匹配不上（实测踩过）。
# 必须带「细则」：《专利法实施细则》第五十条 在原来那份后缀表里
# 只能匹配到「专利法」+「实施细」，于是整句解析失败 —— 而实施细则恰恰
# 是"把母法条文落地"的地方，引用关系最密（出度排第 1 的就是它）。
# 必须带「规则」：《人民检察院民事诉讼监督规则》这类司法解释性文件
# 条文引用关系很密（一个规则里几十条互相援引），原来解析不出来，
# 整句话掉回向量检索 —— 而它的价值恰恰在"第二十四条对应民诉法第几条第几款"。
ARTICLE_REF_RE = re.compile(
    r"[《]?([\u4e00-\u9fa5]{2,18}?(?:法典|法|典|条例|细则|办法|规定|规则|解释))[》]?\s*"
    r"第\s*([〇零一二三四五六七八九十百千两\d]+)\s*条"
)

# 「哪些条文引用了…」「这条引用了哪几条」——问的是**边**，不是条文内容。
# 这类问题决定了图事实要不要前置：本体条文对它是噪声，图事实才是答案。
_CITATION_ASK_RE = re.compile(
    r"(哪些|哪几|什么|多少)(个|条|部)?[^。？?]{0,6}"
    r"(条文|条款|规定|法律|法条)[^。？?]{0,6}(引用|援引|援用|依照|依据|提到|涉及)"
    r"|(引用|援引|援用)了?(哪些|哪几|什么)"
    r"|被(哪些|哪几)[^。？?]{0,4}(条文|条款|条)"
    r"|(与|和)[^。？?]{0,8}(有关|相关)的条文"
)


class Ref(NamedTuple):
    """一条被点名的法条。`law` 是**图里的** law 值（全称），解析不到时为空串。"""

    law: str
    no: str
    raw_law: str


_CN_DIGITS = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def int2cn(n: int) -> str:
    """整数 → 中文条号写法（577 → 五百七十七）。"""
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


def cn2int(s: str) -> int:
    """中文条号 → 整数（五百七十七 → 577）。解析失败返回 0。

    只用于"按条号远近排序"，解析不出就排最后，不影响正确性。
    """
    s = (s or "").strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            cur = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (cur or 1) * _CN_UNITS[ch]
            cur = 0
        else:
            return 0
    return total + cur


def norm_article(num: str) -> str:
    """问句里的条号 → 图里的写法。`510` 与 `五百一十` 要对齐。"""
    num = (num or "").strip()
    if not num:
        return ""
    if num.isdigit():
        return "第" + int2cn(int(num)) + "条"
    return "第" + num.lstrip("第").rstrip("条") + "条" if not num.startswith("第") else num


def asks_citation(text: str) -> bool:
    """这句话问的是"引用关系"吗？

    决定图事实放最前还是最后：问「被哪些条援引」时，本体条文正文是噪声，
    图事实就是答案本身；问「这条怎么规定」时正相反。
    """
    return bool(text) and bool(_CITATION_ASK_RE.search(text))


# ============ 内存索引（懒加载 + TTL + 失败冷却） ============
_lock = threading.Lock()
_loaded_at = 0.0
_failed_at = 0.0
_loaded = False

_laws: List[str] = []
_out: Dict[str, List[dict]] = {}
_in: Dict[str, List[dict]] = {}
_law_refs: Dict[str, List[dict]] = {}


def _key(law: str, no: str) -> str:
    return f"{law}#{no}"


def _load() -> None:
    """把三张表一次拉进内存。任何一条失败都抛，由外层统一降级。

    边只查一次、双向建两份索引：`MATCH (a)-[:CITES]->(b)` 拉一遍，
    正向用 (a,b) 建 out、反向用 (b,a) 建 in。早先分两条查询各写一遍
    MATCH，等于把同一个全表扫描做两遍 —— 装载从 0.14s 涨到 0.27s，
    在云端 2 核上差距更明显。
    """
    global _laws, _out, _in, _law_refs

    laws = [str(r.get("law")) for r in
            (graph.query("MATCH (a:Article) RETURN DISTINCT a.law AS law") or [])
            if r.get("law")]

    edge_rows = graph.query(
        "MATCH (a:Article)-[r:CITES]->(b:Article) "
        "RETURN a.law AS al, a.no AS an, a.chapter AS ach, "
        "b.law AS bl, b.no AS bn, b.chapter AS bch, r.raw AS raw"
    ) or []
    law_rows = graph.query(
        "MATCH (a:Article)-[r:CITES_LAW]->(l:Law) "
        "RETURN a.law AS al, a.no AS an, l.name AS ln, l.in_corpus AS ic, r.raw AS raw"
    ) or []

    out_idx: Dict[str, List[dict]] = {}
    in_idx: Dict[str, List[dict]] = {}
    law_idx: Dict[str, List[dict]] = {}

    for r in edge_rows:
        out_idx.setdefault(_key(r["al"], r["an"]), []).append(
            {"law": r["bl"], "no": r["bn"], "chapter": r.get("bch") or "", "raw": r.get("raw") or ""}
        )
        in_idx.setdefault(_key(r["bl"], r["bn"]), []).append(
            {"law": r["al"], "no": r["an"], "chapter": r.get("ach") or "", "raw": r.get("raw") or ""}
        )

    for r in law_rows:
        # 跨法引用边指向的是 Law 节点，本身没有条号 —— 条号只存在于 raw
        # （"《中华人民共和国个人信息保护法》第三条"）。从 raw 里再解析一次，
        # 才能把"该去看哪一条"落到具体条文上，否则只知道"涉及这部法"。
        raw = r.get("raw") or ""
        m = ARTICLE_REF_RE.search(raw)
        law_idx.setdefault(_key(r["al"], r["an"]), []).append({
            "law": r["ln"],
            "no": norm_article(m.group(2)) if m else "",
            "in_corpus": bool(r.get("ic")),
            "raw": raw,
        })

    _laws, _out, _in, _law_refs = laws, out_idx, in_idx, law_idx


def ensure_loaded(force: bool = False) -> bool:
    """索引是否可用。失败返回 False（30 秒内不再重试），不抛异常。"""
    global _loaded, _loaded_at, _failed_at

    now = time.monotonic()
    if not force and _loaded and now - _loaded_at < _TTL:
        return True
    if not force and now - _failed_at < _FAIL_COOLDOWN:
        return False

    with _lock:
        now = time.monotonic()
        if not force and _loaded and now - _loaded_at < _TTL:
            return True
        if not force and now - _failed_at < _FAIL_COOLDOWN:
            return False
        try:
            _load()
        except Exception as e:  # noqa: BLE001
            _failed_at = time.monotonic()
            logger.warning(f"⚠️ 条文引用图装载失败（{_FAIL_COOLDOWN:.0f}s 内不再重试）: {e}")
            return False
        _loaded, _loaded_at, _failed_at = True, time.monotonic(), 0.0
        logger.info(
            f"📚 条文引用图已装载：{len(_out)} 条出边 / {len(_in)} 条入边 / "
            f"{len(_law_refs)} 条跨法引用 / {len(_laws)} 部法（TTL {_TTL:.0f}s）"
        )
        return True


def resolve_law(name: str) -> List[str]:
    """用户写的法名 → 图里的 law 值。空列表表示图上没有这部法。

    判据与 `law_scope.scoped_sources` 一致（stem 相等优先，其次前缀），
    因为两者最终都要落到同一批文件上：一个决定"查哪几部法的图"，
    另一个决定"召回限定在哪几个文件"。判据不一致会出现
    "图说查商标法、检索却收敛到别处"这种自相矛盾。

    前缀匹配是**故意保留**的：用户写《商标法》时，
    《商标法实施条例》也该在图里一起看 —— 那正是实施细则落地的地方。
    """
    if not name:
        return []
    stem = law_scope.law_stem(name)
    if not stem or not ensure_loaded():
        return []

    exact = [n for n in _laws if law_scope.law_stem(n) == stem]
    if exact:
        return exact

    pref = []
    for n in _laws:
        ns = law_scope.law_stem(n)
        if ns and (ns.startswith(stem) or stem.startswith(ns)):
            pref.append(n)
    # 短的优先：问「商标法」时，`商标法实施条例` 比 `商标法实施条例（2014修订）` 更该先看
    return sorted(pref, key=len)[:3]


def refs_in(text: str) -> List[Ref]:
    """抽出问句里被点名的法条（法名已解析成图里的写法）。"""
    if not text:
        return []
    out: List[Ref] = []
    seen = set()
    for m in ARTICLE_REF_RE.finditer(text):
        raw_law = (m.group(1) or "").strip()
        no = norm_article(m.group(2) or "")
        if not raw_law or not no:
            continue
        for law in (resolve_law(raw_law) or [""]):
            k = (law or raw_law, no)
            if k in seen:
                continue
            seen.add(k)
            out.append(Ref(law=law, no=no, raw_law=raw_law))
    return out


def _pick(rows: List[dict], limit: int) -> List[dict]:
    """邻居条文的取样：跨章轮转，而不是取条号最近的几条。

    为什么不是"条号最近的 N 条"：
      问「510 条适用在哪些情形」，它被援引的地方横跨买卖、租赁、承揽、仓储
      各章，条号最近的那几条全在合同编同一节里 —— 取最近的等于把
      "适用面有多广"这个答案本身抹掉了。按 chapter 轮转，一章一条，
      覆盖的章数最大化。
    """
    if len(rows) <= limit:
        return rows
    by_ch: Dict[str, List[dict]] = {}
    for r in rows:
        by_ch.setdefault(r.get("chapter") or "", []).append(r)
    picked: List[dict] = []
    queues = [q for q in by_ch.values()]
    while len(picked) < limit and any(queues):
        for q in queues:
            if q and len(picked) < limit:
                picked.append(q.pop(0))
        queues = [q for q in queues if q]
    return picked


class Neigh(NamedTuple):
    ref: Ref
    out: List[dict]          # 本条引用了谁（已取样）
    inn: List[dict]          # 谁引用了本条（已取样）
    laws: List[dict]         # 本条涉及的其他法
    total_out: int           # 出边总数（取样前的真实值）
    total_in: int            # 入边总数


def neighbors(ref: Ref, max_out: int = 4, max_in: int = 8) -> Neigh:
    """一跳邻域。图不可用或没这部法时返回空邻域（不是异常）。

    `total_*` 与取样的 `len(*)` 分开返回，是因为**数量本身常常就是答案**：
    问「民法典 510 条被多少条援引」，只报"8 条"（取样上限）是错的，
    真实是 26 条。这个数必须来自全量，不能来自取样结果。
    """
    empty = Neigh(ref, [], [], [], 0, 0)
    if not ref.law or not ensure_loaded():
        return empty
    k = _key(ref.law, ref.no)
    all_out = _out.get(k, [])
    all_in = _in.get(k, [])
    return Neigh(
        ref,
        _pick(all_out, max_out),
        _pick(all_in, max_in),
        _law_refs.get(k, [])[:4],
        len(all_out),
        len(all_in),
    )


def short_law(law: str) -> str:
    """全称 → 简称（去掉「中华人民共和国」），只用于渲染，不用于匹配。"""
    return law_scope.law_stem(law) or law


def _fmt_refs(rows: List[dict]) -> str:
    """「第三十六条、第五十条（第二章）」——同法的合并成一句，跨法的带法名。"""
    if not rows:
        return ""
    same = [r for r in rows if r["law"] == rows[0]["law"]]
    other = [r for r in rows if r["law"] != rows[0]["law"]]
    parts = ["、".join(r["no"] for r in same)] if same else []
    parts += [f"《{short_law(r['law'])}》{r['no']}" for r in other]
    return "、".join(p for p in parts if p)


def facts(refs: List[Ref], max_out: int = 4, max_in: int = 8,
          full: bool = False) -> Tuple[List[str], List[Ref]]:
    """把邻域渲染成确定性事实，同时返回"值得去取原文"的邻居条文。

    返回的邻居条文是**图上确认存在引用关系**的那些 —— 它们不是"语义相近"，
    而是"结构上依赖"。这一步把向量检索捞不到的那批块补进候选池。

    渲染时数量用 `total_*` 而不是取样后的长度：模型会照抄这句话里的数字，
    "被 8 条援引"和真相"被 26 条援引"差了 3 倍，而答案里的数字往往就是考点。

    `full=True`（问句本身在问引用关系时）取消取样上限：这时清单**就是答案**，
    列 4 条却说"另有 1 条未列出"，等于交了个残缺的答案出去。
    取样上限只在"顺带一提"的场景下有意义 —— 那时它控制的是上下文长度。
    """
    if full:
        max_out = max_in = 40
    lines: List[str] = []
    extra: List[Ref] = []
    seen = {(r.law, r.no) for r in refs}

    for ref in refs:
        n = neighbors(ref, max_out=max_out, max_in=max_in)
        if not (n.out or n.inn or n.laws):
            continue
        title = f"《{short_law(ref.law or ref.raw_law)}》{ref.no}"
        bits = []
        if n.total_in:
            more = f"，另有 {n.total_in - len(n.inn)} 条未列出" if n.total_in > len(n.inn) else ""
            bits.append(f"被 {n.total_in} 条援引（{_fmt_refs(n.inn)}{more}）")
        if n.total_out:
            more = f"，另有 {n.total_out - len(n.out)} 条未列出" if n.total_out > len(n.out) else ""
            bits.append(f"引用了 {n.total_out} 条（{_fmt_refs(n.out)}{more}）")
        if n.laws:
            inc = [f"《{short_law(x['law'])}》{x['no']}".strip() for x in n.laws if x["in_corpus"]]
            exc = [f"《{short_law(x['law'])}》{x['no']}".strip() for x in n.laws if not x["in_corpus"]]
            if inc:
                bits.append("另涉" + "、".join(inc) + "（语料内）")
            if exc:
                bits.append("另涉" + "、".join(exc) + "（语料未收录）")
        if bits:
            lines.append(f"{title}：" + "；".join(bits))

        for r in (n.inn + n.out):
            k = (r["law"], r["no"])
            if k not in seen:
                seen.add(k)
                extra.append(Ref(law=r["law"], no=r["no"], raw_law=short_law(r["law"])))
        # 跨法引用落到条号上的那些也能取原文 —— 只有语料里真有这部法时才有意义，
        # 否则 `fetch` 一定 0 块，白花一次 where 查询。
        for x in n.laws:
            if not x["in_corpus"] or not x.get("no"):
                continue
            k = (x["law"], x["no"])
            if k not in seen:
                seen.add(k)
                extra.append(Ref(law=x["law"], no=x["no"], raw_law=short_law(x["law"])))
    return lines, extra


def stats() -> dict:
    """只读状态，不触发加载。给 /health 用。

    与 graph_entities.stats 同样的理由：健康检查不能变成"每次轮询查一次 Neo4j"，
    也不能在 Neo4j 挂掉时陪着等一次超时。要盯的是静默缺失 ——
    loaded=False 意味着所有引用型问题都拿不到图证据，而问答照样返回 200。
    """
    return {
        "loaded": _loaded,
        "out_edges": sum(len(v) for v in _out.values()),
        "in_edges": sum(len(v) for v in _in.values()),
        "cross_law": sum(len(v) for v in _law_refs.values()),
        "laws": len(_laws),
    }


def reset() -> None:
    """清缓存（测试用；改图数据后想立刻生效也可以调）。"""
    global _loaded, _loaded_at, _failed_at, _laws, _out, _in, _law_refs
    with _lock:
        _loaded, _loaded_at, _failed_at = False, 0.0, 0.0
        _laws, _out, _in, _law_refs = [], {}, {}, {}
