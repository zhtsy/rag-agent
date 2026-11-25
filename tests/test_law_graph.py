# -*- coding: utf-8 -*-
"""条文引用图（law_graph）的纯逻辑测试。

不打真库：装载函数靠 monkeypatch 换成内存数据。图的连通性由
`tests/test_law_graph_live.py` 那类联调脚本覆盖，这里只钉住"解析与渲染"
这些出错后**从外部看不出来**的地方：
  · 条号 510 与 五百一十 对不齐 → 图查 0 行 → 退化成向量检索，答案看起来"还行"
  · 数量用取样长度而不是总数 → 模型照抄"被 8 条援引"，真值是 26
"""
import pytest

import law_graph as lg


class _FakeGraph:
    """只回答 _load 里那三条 Cypher。"""

    def __init__(self, laws, edges, law_edges):
        self.laws, self.edges, self.law_edges = laws, edges, law_edges

    def query(self, cypher, *a, **kw):
        if "DISTINCT a.law" in cypher:
            return [{"law": x} for x in self.laws]
        if "CITES_LAW" in cypher:
            return self.law_edges
        if "CITES]->" in cypher:      # 边查询写的是 -[r:CITES]->，不是 [:CITES]
            return self.edges
        return []


def _edge(a_law, a_no, b_law, b_no, a_ch="", b_ch="", raw=""):
    return {"al": a_law, "an": a_no, "ach": a_ch, "bl": b_law, "bn": b_no, "bch": b_ch, "raw": raw}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    laws = ["中华人民共和国商标法", "中华人民共和国商标法实施条例", "中华人民共和国民法典"]
    edges = [
        _edge("中华人民共和国商标法", "第三十六条", "中华人民共和国商标法", "第十五条", "第七章", "第二章"),
        _edge("中华人民共和国商标法", "第五十条", "中华人民共和国商标法", "第十五条", "第七章", "第二章"),
        _edge("中华人民共和国商标法", "第五十四条", "中华人民共和国商标法", "第十五条", "第七章", "第二章"),
    ] + [
        _edge("中华人民共和国民法典", f"第{n}条", "中华人民共和国民法典", "第五百一十条", f"第{n//100}章", "第三章")
        for n in range(582, 610)
    ]
    law_edges = [
        # 语料未收录：只能提示"另涉"，取不到原文
        {"al": "中华人民共和国商标法", "an": "第十五条",
         "ln": "中华人民共和国反不正当竞争法", "ic": False,
         "raw": "《中华人民共和国反不正当竞争法》第六条"},
        # 语料内：条号应被解析出来并进 extra，供 _fetch_by_refs 取原文
        {"al": "中华人民共和国商标法", "an": "第十五条",
         "ln": "中华人民共和国消费者权益保护法", "ic": True,
         "raw": "《中华人民共和国消费者权益保护法》第八条"},
    ]
    monkeypatch.setattr(lg, "graph", _FakeGraph(laws, edges, law_edges))
    lg.reset()
    yield
    lg.reset()


def test_int2cn_roundtrip():
    for n in (5, 23, 36, 510, 577, 1260):
        assert lg.cn2int(lg.int2cn(n)) == n


def test_norm_article_aligns_arabic_and_chinese():
    assert lg.norm_article("510") == "第五百一十条"
    assert lg.norm_article("五百一十") == "第五百一十条"
    assert lg.norm_article("第十五条") == "第十五条"


def test_ref_regex_handles_shixize_and_full_name():
    # 「实施细则」不在原来的后缀表里，整句会解析失败（实测踩过）
    m = lg.ARTICLE_REF_RE.search("《专利法实施细则》第五十条引用了哪些条文")
    assert m and m.group(1) == "专利法实施细则"
    m = lg.ARTICLE_REF_RE.search("《中华人民共和国商标法》第十五条")
    assert m and m.group(1) == "中华人民共和国商标法"


def test_resolve_law_prefers_exact_stem(fake):
    # 问「商标法」时，实施条例是前缀匹配的候选，但精确相等的那部必须排第一
    got = lg.resolve_law("商标法")
    assert got[0] == "中华人民共和国商标法"
    assert lg.resolve_law("民法典") == ["中华人民共和国民法典"]
    assert lg.resolve_law("压根不存在的法") == []


def test_refs_in_resolves_to_graph_law_name(fake):
    refs = lg.refs_in("商标法第15条被哪些条文引用")
    assert refs and refs[0].law == "中华人民共和国商标法"
    assert refs[0].no == "第十五条"


def test_asks_citation_distinguishes_relation_from_content():
    assert lg.asks_citation("《商标法》第十五条被哪些条文引用？")
    assert lg.asks_citation("专利法实施细则第五十条引用了哪些条文")
    assert not lg.asks_citation("《专利法》第三十六条是怎么规定的？")
    assert not lg.asks_citation("商标法第十五条的内容是什么")


def test_neighbors_direction(fake):
    ref = lg.Ref(law="中华人民共和国商标法", no="第十五条", raw_law="商标法")
    n = lg.neighbors(ref)
    assert n.total_in == 3 and n.total_out == 0
    assert {r["no"] for r in n.inn} == {"第三十六条", "第五十条", "第五十四条"}


def test_facts_reports_true_total_not_sample_size(fake):
    """取样 3 条时总数是 3；取样被截到 2 条时，文案必须仍报 26。"""
    ref = lg.Ref(law="中华人民共和国民法典", no="第五百一十条", raw_law="民法典")
    lines, extra = lg.facts([ref], max_in=2)
    assert len(lines) == 1
    assert "被 28 条援引" in lines[0]
    assert "另有 26 条未列出" in lines[0]
    assert len([e for e in extra]) == 2


def test_facts_expands_cross_law_reference_to_article(fake):
    """跨法引用边的条号藏在 raw 里；只有语料内的那部才值得去取原文。"""
    ref = lg.Ref(law="中华人民共和国商标法", no="第十五条", raw_law="商标法")
    lines, extra = lg.facts([ref])
    assert any("反不正当竞争法》第六条（语料未收录）" in ln for ln in lines)
    assert any("消费者权益保护法》第八条（语料内）" in ln for ln in lines)
    assert [e.no for e in extra if e.law.endswith("消费者权益保护法")] == ["第八条"]
    assert not [e for e in extra if e.law.endswith("反不正当竞争法")]


def test_unavailable_graph_returns_empty_without_raising(monkeypatch):
    class _Boom:
        def query(self, *a, **kw):
            raise RuntimeError("Neo4j down")

    monkeypatch.setattr(lg, "graph", _Boom())
    lg.reset()
    assert lg.ensure_loaded() is False
    ref = lg.Ref(law="中华人民共和国商标法", no="第十五条", raw_law="商标法")
    assert lg.neighbors(ref).total_in == 0
    assert lg.facts([ref]) == ([], [])
    lg.reset()


def test_stats_does_not_trigger_load(monkeypatch):
    """健康检查不能变成"每次轮询查一次 Neo4j"。"""
    called = []

    class _Spy:
        def query(self, *a, **kw):
            called.append(1)
            return []

    monkeypatch.setattr(lg, "graph", _Spy())
    lg.reset()
    s = lg.stats()
    assert called == []
    assert s["loaded"] is False
