# -*- coding: utf-8 -*-
"""补齐宠物节点的「品种是」边（依据原文，可逆）。

背景：
    图谱里 4 个宠物只有 2 个带类型边（布丁→仓鼠、豆豆→金毛）。
    「代码」「拿铁」建图时漏抽了类型，导致答案只能说"小李养了代码"，
    说不出"是一只橘猫"——而原文写得很清楚。

用法：
    python fix_pet_types.py            # dry-run，只打印将做的改动
    python fix_pet_types.py --apply    # 真正写库

回滚：
    MATCH (p {id:'代码'})-[r:品种是]->() DELETE r
    MATCH (p {id:'拿铁'})-[r:品种是]->() DELETE r
    MATCH (k) WHERE k.id IN ['橘猫'] AND NOT (k)<-[]-() DELETE k
    （「猫」是通用节点，可能本来就有，回滚时不要删）
"""
import sys

from neo4j import GraphDatabase

# (宠物, 品种, 来源文件, 原文依据)
FIXES = [
    ("代码", "橘猫", "test5.txt", "我养了一只橘猫，叫“代码”，是我两年前从小区楼下捡回来的流浪猫"),
    ("拿铁", "猫", "test10.txt", "我养了一只叫“拿铁”的猫，是我在楼下捡的"),
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
        # 品种节点没有就建，并打上来源，保证答案能引用到文件
        s.run(
            "MERGE (k {id:$kind}) SET k.source = coalesce(k.source, $src)",
            kind=kind, src=src,
        )
        s.run(
            "MATCH (p {id:$pet}), (k {id:$kind}) "
            "MERGE (p)-[r:品种是]->(k) "
            "RETURN type(r) AS t",
            pet=pet, kind=kind,
        )
        check = s.run(
            "MATCH (p {id:$pet})-[r:品种是]->(k) RETURN k.id AS k", pet=pet
        ).data()
        print("  -> 已写入，校验: %s\n" % [r["k"] for r in check])
    else:
        print("  -> 将执行: MERGE (:%s {id:'%s'}) SET source='%s'" % (kind, kind, src))
        print("  -> 将执行: (:%s)-[:品种是]->(:%s)\n" % (pet, kind))

print("== 补完后全部 养了 关系 ==")
for r in s.run(
    "MATCH (o)-[:养了]->(p) OPTIONAL MATCH (p)-[:品种是]->(k) "
    "RETURN o.id AS owner, p.id AS pet, collect(k.id) AS kinds"
).data():
    print("  %s 养了 %s -> %s" % (r["owner"], r["pet"], r["kinds"] or "无"))
d.close()
