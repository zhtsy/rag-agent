# -*- coding: utf-8 -*-
"""补齐宠物节点的「品种是」边（依据原文，可逆）。

背景：
    `graph_retriever._TYPE_RELS` 只认 ("品种是", "类型是", "职位是")，
    故意不含 `属于`——因为 `属于` 在图谱里被复用得很宽
    （小刘→校篮球队、片儿川→杭州、虾饺→广式早茶…），
    拿它当"类型"会把答案污染成胡说。

    但建图时宠物品种恰恰被抽成了 `属于`：
        布丁 -[属于]-> 仓鼠
        豆豆 -[属于]-> 金毛
        代码 -[属于]-> 橘猫
    而「拿铁」连出边都没有。

    结果就是补品种那一跳永远补不上，答案只能说"小李养了代码"，
    说不出"是一只橘猫"——原文写得很清楚。

    所以修法在数据侧：补 `品种是` 边，不动 `_TYPE_RELS` 白名单。

用法：
    python fix_pet_types.py            # dry-run，只打印将做的改动
    python fix_pet_types.py --apply    # 真正写库

回滚：
    补边前图谱里 `品种是` 边为 0 条，所以回滚可以整类型删除：
        MATCH ()-[r:品种是]->() DELETE r
    精确回滚单条：
        MATCH (p {id:$pet})-[r:品种是]->() DELETE r
"""
import sys

from neo4j import GraphDatabase

# (宠物, 品种, 来源文件, 原文依据)
FIXES = [
    ("代码", "橘猫", "test5.txt", "我养了一只橘猫，叫“代码”，是我两年前从小区楼下捡回来的流浪猫"),
    ("拿铁", "猫", "test10.txt", "我养了一只叫“拿铁”的猫，是我在楼下捡的"),
    ("布丁", "仓鼠", "test6.txt", "阳台上养了几盆多肉，还有一只叫“布丁”的仓鼠"),
    ("豆豆", "金毛", "test8.txt", "小陈养了一只金毛叫豆豆"),
]

APPLY = "--apply" in sys.argv

from settings import neo4j_kwargs

_kw = neo4j_kwargs()
d = GraphDatabase.driver(_kw["url"], auth=(_kw["username"], _kw["password"]))
s = d.session()

print("模式:", "APPLY（写库）" if APPLY else "DRY-RUN（只读）")
print()

for pet, kind, src, evidence in FIXES:
    has_pet = s.run("MATCH (n {id:$i}) RETURN count(n) AS c", i=pet).single()["c"]
    has_kind = s.run("MATCH (n {id:$i}) RETURN count(n) AS c", i=kind).single()["c"]
    existing = s.run(
        "MATCH (p {id:$i})-[r:品种是]->(k) RETURN k.id AS k", i=pet
    ).data()

    print("【%s】" % pet)
    print("  宠物节点存在: %d ｜ 品种节点「%s」存在: %d" % (has_pet, kind, has_kind))
    print("  现有品种边: %s" % ([r["k"] for r in existing] or "无"))
    print("  原文: %s" % evidence)

    if existing:
        print("  -> 已有品种边，跳过\n")
        continue
    if not has_pet:
        print("  -> 宠物节点不存在，跳过（需人工确认）\n")
        continue

    if APPLY:
        # 品种节点不存在才建，并打上来源；已存在则只补边，不动其属性
        # （「猫」是通用类别节点、没有 source，不该被某一篇文档"认领"）
        if has_kind == 0:
            s.run(
                "MERGE (k {id:$kind}) SET k.source = coalesce(k.source, $src)",
                kind=kind, src=src,
            )
            print("  -> 新建品种节点「%s」source=%s" % (kind, src))
        else:
            print("  -> 品种节点「%s」已存在，只补边" % kind)
        s.run(
            "MATCH (p {id:$pet}), (k {id:$kind}) MERGE (p)-[r:品种是]->(k)",
            pet=pet, kind=kind,
        )
        check = s.run(
            "MATCH (p {id:$pet})-[r:品种是]->(k) RETURN k.id AS k", pet=pet
        ).data()
        print("  -> 已写入，校验: %s\n" % [r["k"] for r in check])
    else:
        print("  -> 将执行: MERGE (:%s {id:'%s'}) SET source='%s'" % (kind, kind, src))
        print("  -> 将执行: (:%s)-[:品种是]->(:%s)\n" % (pet, kind))

print("== 补完后，按 graph_retriever 的两跳模板验证 ==")
print("   (人物)-[:养了]->(pet)-[:品种是|属于*0..1]->(b)-[:属于类别*0..1]->(cat)")
for r in s.run(
    "MATCH (o:人物)-[:养了]->(p)-[:品种是|属于*0..1]->(b)-[:属于类别*0..1]->(cat) "
    "RETURN o.id AS owner, p.id AS pet, b.id AS mid, cat.id AS terminal "
    "ORDER BY owner, pet"
).data():
    print("  %-4s 养了 %-4s -> %-4s -> %s" % (r["owner"], r["pet"], r["mid"], r["terminal"]))
d.close()
