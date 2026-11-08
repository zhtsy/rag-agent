"""地理层级：城市 → 省级行政区的映射，以及补边计划的计算。

为什么需要这个模块
------------------
图谱抽取只认「文本里共现的词」。原文写「我住在南京」，就只有 `(小智)-[:住在]->(南京)`，
没有「江苏」二字，所以 `(南京)-[:属于]->(江苏)` 从来不会被抽出来。

而 `graph_retriever.CYPHER_GENERATION_TEMPLATE` 的规则 2 写着：

    如果问题涉及"省份"、"省"，必须使用两跳查询：人物 -[:住在]-> 城市 -[:属于]-> 省份

也就是说 **查询层的契约早就写好了，数据层从来没有满足过它**。
于是「小陈住在哪个省」这道题，Cypher 生成得完全正确，执行返回 0 行，
模型只能如实说「资料里没有说明」。

这不是常识缺失，是 schema 缺一层。判据：扬州有边，南京没有 —— 差别不在知识，
在于原文有没有把"江苏"和"扬州"写在一起。

设计上的三个决定
----------------
1. **复用 `属于`，不新建关系类型。**
   `属于` 在图上已经混装了多种语义（`融合菜→川菜` 菜系、`小刘→校篮球队` 组织、
   `珍珠奶茶→奶茶` 分类）。再开一个 `位于` 会让查询模板、关系白名单、Cypher 提示词
   三处都要改，而收益只是"语义更干净"——这是审美收益，不是功能收益。
   代价用**属性标记**补回来：补出来的边带 `origin='geo_backfill'`，随时可查可回滚，
   也能和抽取出来的边分开统计。见 `audit_cypher()`。

2. **直辖市建自环。**
   北京/上海/重庆/天津本身就是省级行政区。问「小张住在哪个省」，答案是"上海"。
   若不建自环，这类问题必然查不到。自环 `(上海)-[:属于]->(上海)` 在 Cypher 里
   完全合法，两跳会返回自身节点，答案正确。标记 `self_ref=True` 便于审计。

3. **只补有节点的。**
   映射表故意写得比当前图谱大（覆盖主要城市），但补边时 `MATCH` 不到节点就跳过并
   计入 skipped —— 绝不 `MERGE` 创建空节点。唯一例外是省级节点缺失时显式新建
   （如「浙江」），且必须与 `build_graph.write_graph` 的格式一致：
   `MERGE (n:__Entity__ {id}) SET n:地点`。

零第三方依赖：纯数据 + 纯函数，可直接进 CI。
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple

# ========== 1. 直辖市 ==========
# 这四个的"省"就是自己。判定自环靠这个集合，不要靠字符串比较。
MUNICIPALITIES: Set[str] = {"北京", "上海", "天津", "重庆"}

# ========== 2. 城市 → 省级行政区 ==========
# 覆盖当前图谱里出现过的全部中国地理实体（含区级），外加常见城市以备后续入库。
# 键用节点 `id` 原样，不改写图谱里的既有命名（如"深圳南山区"保持带区）。
CITY_TO_PROVINCE: Dict[str, str] = {
    # —— 当前图谱里真实存在的 ——
    "南京": "江苏",
    "扬州": "江苏",
    "广州": "广东",
    "深圳": "广东",
    "深圳南山区": "广东",
    "潮汕地区": "广东",
    "河源万绿湖": "广东",
    "成都": "四川",
    "彭州": "四川",
    "西安": "陕西",
    "西安竹笆市": "陕西",
    "杭州": "浙江",
    "兰州": "甘肃",
    "北京": "北京",
    "北京城": "北京",
    "上海": "上海",
    "上海静安区": "上海",
    "重庆": "重庆",
    "重庆嘉陵江畔": "重庆",
    # —— 尚未入库，预置以防将来新增文档 ——
    "苏州": "江苏",
    "无锡": "江苏",
    "常州": "江苏",
    "徐州": "江苏",
    "南通": "江苏",
    "珠海": "广东",
    "佛山": "广东",
    "东莞": "广东",
    "中山": "广东",
    "惠州": "广东",
    "汕头": "广东",
    "宁波": "浙江",
    "温州": "浙江",
    "绍兴": "浙江",
    "嘉兴": "浙江",
    "金华": "浙江",
    "绵阳": "四川",
    "德阳": "四川",
    "宜宾": "四川",
    "乐山": "四川",
    "宝鸡": "陕西",
    "咸阳": "陕西",
    "榆林": "陕西",
    "天津": "天津",
    "长沙": "湖南",
    "武汉": "湖北",
    "郑州": "河南",
    "济南": "山东",
    "青岛": "山东",
    "合肥": "安徽",
    "福州": "福建",
    "厦门": "福建",
    "南昌": "江西",
    "太原": "山西",
    "石家庄": "河北",
    "沈阳": "辽宁",
    "大连": "辽宁",
    "长春": "吉林",
    "哈尔滨": "黑龙江",
    "昆明": "云南",
    "贵阳": "贵州",
    "南宁": "广西",
    "海口": "海南",
    "呼和浩特": "内蒙古",
    "银川": "宁夏",
    "西宁": "青海",
    "乌鲁木齐": "新疆",
    "拉萨": "西藏",
}

# ========== 3. 已经是省级行政区的名字 ==========
# 用作「补边时不要给省节点再找省」的护栏，也用于校验映射表的方向。
PROVINCE_NAMES: Set[str] = {
    "北京", "天津", "上海", "重庆",                                   # 直辖市
    "河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽",
    "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "海南",
    "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾",
    "内蒙古", "广西", "西藏", "宁夏", "新疆",                          # 自治区
    "香港", "澳门",                                                   # 特别行政区
}

# 补出来的边统一打这个标记，便于审计与回滚。
BACKFILL_ORIGIN = "geo_backfill"


class GeoEdge:
    """一条待补的边。用类而不是 dict，是为了让 `is_self_ref` 无法被写成字符串。"""

    __slots__ = ("city", "province", "is_self_ref")

    def __init__(self, city: str, province: str) -> None:
        self.city = city
        self.province = province
        self.is_self_ref = city == province

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        kind = "自环" if self.is_self_ref else "普通"
        return f"GeoEdge({self.city} -> {self.province}, {kind})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, GeoEdge)
                and self.city == other.city
                and self.province == other.province)

    def __hash__(self) -> int:
        return hash((self.city, self.province))


def resolve_province(city: str) -> Optional[str]:
    """城市名 → 省名。查不到返回 None（不猜、不模糊匹配）。"""
    if not city:
        return None
    return CITY_TO_PROVINCE.get(city.strip())


def plan_edges(existing_nodes: Iterable[str]) -> Tuple[List[GeoEdge], List[str]]:
    """算出该补哪些边。

    参数
    ----
    existing_nodes : 图谱里现有的全部节点 id

    返回
    ----
    (edges, missing_provinces)
        edges              —— 两端节点都存在的边，可直接 MERGE
        missing_provinces  —— 城市存在但省节点缺失，需先建节点（去重、保序）

    为什么分两批返回：建节点和建边是两种风险。建节点会改变节点总数，必须让人看见；
    建边只改变边数。混在一起做，出问题时无法判断是哪一步动的。
    """
    nodes = set(existing_nodes)
    edges: List[GeoEdge] = []
    missing: List[str] = []
    seen_missing: Set[str] = set()

    for city, province in CITY_TO_PROVINCE.items():
        if city not in nodes:
            continue                        # 图里没这个城市，跳过（绝不凭空造节点）
        if province in nodes:
            edges.append(GeoEdge(city, province))
        else:
            # 城市在、省不在：省名必须真的是个省级行政区，避免映射表写错导致乱建节点
            if province not in PROVINCE_NAMES:
                continue
            edges.append(GeoEdge(city, province))
            if province not in seen_missing:
                seen_missing.add(province)
                missing.append(province)

    return edges, missing


def audit_cypher() -> str:
    """审计用查询：把"抽取来的边"和"补出来的边"分开数。

    这条语句存在的意义：`属于` 混装了多种语义，一旦有人质疑"这些边是哪来的"，
    需要能一句话答出来，而不是翻脚本。
    """
    return (
        "MATCH (a)-[r:属于]->(b) "
        "RETURN coalesce(r.origin, 'extracted') AS origin, "
        "       count(*) AS edges, "
        "       collect(a.id + ' -> ' + b.id)[0..5] AS sample "
        "ORDER BY edges DESC"
    )
