"""地理层级映射的单元测试。

这组测试守的是**不变量**，不是具体某几条映射。理由：
映射表是手写的，最容易出的错不是"漏了一个城市"（漏了只是补不上，不影响正确性），
而是**方向写反**（把"江苏"映射到"南京"）——那种错会静默污染图谱，
让「小陈住在哪个省」查出一个城市名。

全程零第三方依赖，可进 0.2 秒 CI。
"""

import pytest

from geo_hierarchy import (
    CITY_TO_PROVINCE,
    MUNICIPALITIES,
    PROVINCE_NAMES,
    GeoEdge,
    audit_cypher,
    plan_edges,
    resolve_province,
)


# ----------------------------------------------------------------------
# 1. 映射表自身的一致性
# ----------------------------------------------------------------------

def test_every_province_value_is_a_real_province():
    """所有映射目标必须是真的省级行政区名。

    防的是拼写错误：把"广东"写成"广东省"，图谱里就查不到，
    补边会静默失败（MATCH 不到），而且不会报错。
    """
    bad = {c: p for c, p in CITY_TO_PROVINCE.items() if p not in PROVINCE_NAMES}
    assert not bad, f"这些映射目标不是合法省名: {bad}"


def test_no_province_is_used_as_a_city_key():
    """省名不该出现在 key 位——除了直辖市（它们的"省"就是自己）。

    方向写反是这类表最危险的错误：`("江苏", "南京")` 会让
    `(江苏)-[:属于]->(南京)`，查询返回"南京"，答案变成一个城市。
    """
    bad = [c for c in CITY_TO_PROVINCE if c in PROVINCE_NAMES and c not in MUNICIPALITIES]
    assert not bad, f"这些省名被错当成城市: {bad}"


def test_self_reference_only_for_municipalities():
    """自环只能给直辖市。普通城市自环说明映射表填错了。"""
    self_refs = {c for c, p in CITY_TO_PROVINCE.items() if c == p}
    assert self_refs <= MUNICIPALITIES, f"非直辖市出现自环: {self_refs - MUNICIPALITIES}"


def test_municipalities_present_in_mapping():
    """直辖市必须在映射里，否则"住在哪个省"对北京/上海必然查不到。"""
    for m in MUNICIPALITIES & {"北京", "上海", "重庆"}:
        assert CITY_TO_PROVINCE.get(m) == m


# ----------------------------------------------------------------------
# 2. resolve_province 的行为
# ----------------------------------------------------------------------

@pytest.mark.parametrize("city,expected", [
    ("南京", "江苏"),
    ("广州", "广东"),
    ("深圳", "广东"),
    ("深圳南山区", "广东"),   # 区级也要能直接映射到省
    ("成都", "四川"),
    ("西安", "陕西"),
    ("杭州", "浙江"),
    ("北京", "北京"),
])
def test_resolve_known_cities(city, expected):
    assert resolve_province(city) == expected


@pytest.mark.parametrize("city", ["", None, "瓦洛兰", "泰拉大陆", "老家", "某不存在的地方"])
def test_resolve_unknown_returns_none(city):
    """查不到就返回 None，绝不模糊匹配。

    图里有一批虚构地名（瓦洛兰、泰拉大陆是游戏地图），
    模糊匹配会把它们误判成真实城市，那种错比查不到严重得多。
    """
    assert resolve_province(city) is None


def test_resolve_strips_whitespace():
    assert resolve_province("  南京 ") == "江苏"


# ----------------------------------------------------------------------
# 3. plan_edges 的行为
# ----------------------------------------------------------------------

def test_skips_cities_not_in_graph():
    """图里没有的城市不补边——避免 MERGE 造出空节点。

    映射表故意写得比图谱大，所以这条是常态路径，不是边缘情况。
    """
    edges, missing = plan_edges(["江苏", "广东"])       # 没有任何城市
    assert edges == []
    assert missing == []


def test_produces_edge_when_both_ends_exist():
    edges, missing = plan_edges(["南京", "江苏", "广州", "广东"])
    got = {(e.city, e.province) for e in edges}
    assert ("南京", "江苏") in got
    assert ("广州", "广东") in got
    assert missing == []


def test_reports_missing_province_node():
    """省节点缺失时要显式报出来，让执行层先建节点。

    不报的话，`MATCH (p {id:'浙江'})` 返回 0 行，MERGE 静默跳过，
    看起来"补了"实际上没有——这是最典型的假成功。
    """
    edges, missing = plan_edges(["杭州"])
    assert missing == ["浙江"]
    assert any(e.city == "杭州" and e.province == "浙江" for e in edges)


def test_missing_province_deduplicated():
    """多个城市共用一个缺失的省时，只建一次节点。"""
    edges, missing = plan_edges(["杭州", "宁波", "温州"])
    assert missing == ["浙江"]
    assert len(edges) == 3


def test_ignores_invalid_province_target():
    """映射目标不是合法省名时直接跳过，不建节点。

    这条防的是"映射表被改错"的连带伤害：错了不该造出一个假省节点。
    """
    edges, missing = plan_edges(["南京"])
    assert all(e.province != "南京" for e in edges)


def test_plan_is_deterministic():
    """同样输入必须给同样输出（dict 保序），便于 diff 与复现。"""
    nodes = ["南京", "江苏", "广州", "广东", "杭州"]
    assert plan_edges(nodes) == plan_edges(nodes)


def test_plan_covers_known_people_cities():
    """回归：这几个城市对应评测里的实际题目，缺一个就会掉分。

    T031 = 小陈（广州）、T032 = 小智（南京）。
    """
    nodes = ["南京", "江苏", "广州", "广东", "深圳", "成都", "四川",
             "西安", "陕西", "杭州", "北京", "上海", "重庆"]
    edges, _ = plan_edges(nodes)
    pairs = {(e.city, e.province) for e in edges}
    for must in [("南京", "江苏"), ("广州", "广东"), ("深圳", "广东"),
                 ("成都", "四川"), ("西安", "陕西"), ("杭州", "浙江")]:
        assert must in pairs, f"缺少关键映射: {must}"


# ----------------------------------------------------------------------
# 4. GeoEdge 的语义
# ----------------------------------------------------------------------

def test_geoedge_self_ref_flag():
    assert GeoEdge("上海", "上海").is_self_ref is True
    assert GeoEdge("南京", "江苏").is_self_ref is False


def test_geoedge_equality():
    assert GeoEdge("南京", "江苏") == GeoEdge("南京", "江苏")
    assert GeoEdge("南京", "江苏") != GeoEdge("南京", "浙江")


# ----------------------------------------------------------------------
# 5. 审计查询
# ----------------------------------------------------------------------

def test_audit_cypher_is_read_only():
    """审计语句必须只读——它会被运维随手执行，不能有写关键字。"""
    q = audit_cypher()
    assert q.strip().upper().startswith("MATCH")
    for kw in ("CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP", "CALL"):
        assert kw not in q.upper(), f"审计语句里出现了 {kw}"
