# -*- coding: utf-8 -*-
"""品种 → 类别的常识层级，用来把「养猫的是谁」这类聚合问法变成一次确定查询。

为什么需要
----------
图谱里 `(宠物)-[:品种是]->(品种)` 只到品种这一层：

    代码 -[:品种是]-> 橘猫
    拿铁 -[:品种是]-> 猫
    豆豆 -[:品种是]-> 金毛

所以 `WHERE 品种 = "猫"` 只能查到拿铁，**会漏掉橘猫**——而漏掉一个人的聚合答案，
比检索落空更糟：它会理直气壮地给出一个不完整的名单。
这和地理层级是同一类问题（`(城市)-[:属于]->(省)` 补之前，"扬州属于哪个省"也答不出）。

为什么不用 `属于`
----------------
`属于` 在图谱里被复用得极宽（小刘属于校篮球队、片儿川属于杭州、珍珠奶茶属于奶茶），
`graph_retriever.py` 顶部专门写了"绝对不要拿它当类型"。再往里塞层级只会更乱。
所以这里用**独立关系** `属于类别`，语义单一，也便于整体回滚。

用法
----
    from breed_hierarchy import category_of, plan_edges

    category_of("橘猫")      # "猫"
    category_of("哈士奇")    # "狗"
    category_of("史莱姆")    # None（不是已知品种，不猜）
    plan_edges(node_ids)     # 只给"两端节点都在图里"的边
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

# 关系名。刻意不复用 `属于`，理由见模块文档。
CATEGORY_REL = "属于类别"

# 补出来的边的来源标记（回滚依据）
BREED_BACKFILL_ORIGIN = "breed_backfill"

# 新建类别节点用的标签。图谱里原本没有这一层，凭空加标签是为了让 Cypher 能写
# `(b)-[:属于类别*0..1]->(cat:类别)` 而不必靠 WHERE 猜。
CATEGORY_LABEL = "类别"


def _norm(s: object) -> str:
    """统一比较口径：去空白、去全角空格、casefold。"""
    if s is None:
        return ""
    return str(s).replace("　", " ").strip().casefold()


# 品种 → 类别。中文宠物常见叫法，故意写全一点：
# 这份表是"常识"，不是从语料里抽的，所以宁可多几条无用的，也别在语料出现新宠物时改代码。
BREED_TO_CATEGORY: dict = {
    # ---- 猫 ----
    "橘猫": "猫",
    "狸花猫": "猫",
    "三花猫": "猫",
    "奶牛猫": "猫",
    "黑猫": "猫",
    "白猫": "猫",
    "中华田园猫": "猫",
    "英短": "猫",
    "英国短毛猫": "猫",
    "美短": "猫",
    "美国短毛猫": "猫",
    "布偶": "猫",
    "布偶猫": "猫",
    "暹罗": "猫",
    "暹罗猫": "猫",
    "波斯猫": "猫",
    "加菲猫": "猫",
    "折耳猫": "猫",
    "蓝猫": "猫",
    "银渐层": "猫",
    "金渐层": "猫",
    "缅因猫": "猫",
    "无毛猫": "猫",
    "小猫": "猫",
    "猫咪": "猫",
    # ---- 狗 ----
    "金毛": "狗",
    "金毛寻回犬": "狗",
    "哈士奇": "狗",
    "柯基": "狗",
    "泰迪": "狗",
    "柴犬": "狗",
    "拉布拉多": "狗",
    "边牧": "狗",
    "边境牧羊犬": "狗",
    "萨摩耶": "狗",
    "博美": "狗",
    "比熊": "狗",
    "德牧": "狗",
    "德国牧羊犬": "狗",
    "中华田园犬": "狗",
    "小狗": "狗",
    "狗狗": "狗",
    # ---- 其他 ----
    "仓鼠": "仓鼠",
    "金丝熊": "仓鼠",
    "兔子": "兔",
    "垂耳兔": "兔",
    "龙猫": "龙猫",
    "鹦鹉": "鸟",
    "玄凤": "鸟",
    "乌龟": "龟",
    "草龟": "龟",
    "金鱼": "鱼",
    "锦鲤": "鱼",
}

# 合法类别集合，供不变量测试使用。
CATEGORY_NAMES = frozenset(BREED_TO_CATEGORY.values())

_NORM_MAP = {_norm(k): v for k, v in BREED_TO_CATEGORY.items()}


def category_of(breed: object) -> str | None:
    """品种 → 类别；不认识就返回 None（不猜、不兜底成"宠物"）。"""
    key = _norm(breed)
    if not key:
        return None
    return _NORM_MAP.get(key)


@dataclass(frozen=True)
class Edge:
    """一条待补的层级边。"""

    breed: str
    category: str

    @property
    def is_self_ref(self) -> bool:
        """品种本身就是类别（拿铁的品种是"猫"）：不该补自环。"""
        return _norm(self.breed) == _norm(self.category)

    def __str__(self) -> str:  # pragma: no cover - 只为打印
        return f"({self.breed})-[:{CATEGORY_REL}]->({self.category})"


def plan_edges(node_ids: Iterable[object]) -> Tuple[List[Edge], List[str]]:
    """给出待补的边，以及需要新建的类别节点。

    两条不变量：
      1. **只补两端都在图里的边。** 品种节点不存在时，补出来的边挂不到东西上，
         而且会凭空造节点，污染"节点数"这类审计指标。
      2. **不补自环。** 拿铁的品种就是"猫"，再补 `猫-[:属于类别]->猫` 只会在
         变长路径里制造无意义的循环。

    返回 (edges, missing_categories)。
    """
    nodes = {_norm(n) for n in node_ids if _norm(n)}
    display = {}
    for n in node_ids:
        k = _norm(n)
        if k and k not in display:
            display[k] = str(n).strip()

    edges: List[Edge] = []
    for breed, category in BREED_TO_CATEGORY.items():
        bk, ck = _norm(breed), _norm(category)
        if bk not in nodes:
            continue
        e = Edge(breed=display.get(bk, breed), category=category)
        if e.is_self_ref:
            continue
        edges.append(e)

    missing = sorted({e.category for e in edges if _norm(e.category) not in nodes})
    return edges, missing


def audit_cypher() -> str:
    """按来源统计 `属于类别` 边，用于补边后核对。

    刻意写成 `MATCH ()-[r]->() WHERE type(r) = ...` 而不是 `-[r:属于类别]->`：
    关系类型还不存在时，后者会让 Neo4j 抛 01N51 告警刷屏（第一次跑尤其明显）。
    """
    return (
        "MATCH ()-[r]->() WHERE type(r) = '"
        + CATEGORY_REL
        + "' RETURN coalesce(r.origin, 'extracted') AS origin, count(r) AS edges, "
        "head(collect(startNode(r).id)) AS sample ORDER BY edges DESC"
    )


# 供测试与文档引用：映射表的规模
def summary() -> Sequence[Tuple[str, int]]:
    counts = {}
    for cat in BREED_TO_CATEGORY.values():
        counts[cat] = counts.get(cat, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])
