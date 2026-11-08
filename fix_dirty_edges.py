# -*- coding: utf-8 -*-
"""清理抽取错误的图谱边（可逆，先备份再执行）。

问题边：
    1. 小李 -[:养了]-> 橘猫
       原文「我养了一只橘猫，叫"代码"」——只有一只猫，橘猫是品种不是第二只宠物。
       这条边让事实句变成"小李养了橘猫、猫、代码"，模型只能猜哪个是名字。
       特征：无 chunk 属性（不是从原文块抽取的，是后期误补）。

    2. 代码 -[:住在]-> 小区楼下
       原文「是我两年前从小区楼下捡回来的流浪猫」——不是"住在"。
       特征：无 chunk 属性。

用法：
    python fix_dirty_edges.py            # dry-run，只打印
    python fix_dirty_edges.py --apply    # 备份 JSON 后删除

回滚：
    见 /tmp/dirty_edges_backup.json，按里面记录的 (src, rel, dst)
    逐条 MERGE 即可恢复。
"""
import json
import sys

from neo4j import GraphDatabase

from settings import neo4j_kwargs

APPLY = "--apply" in sys.argv
BACKUP = "/tmp/dirty_edges_backup.json"

# (起点 id, 关系类型, 终点 id, 判定理由)
TARGETS = [
    ("小李", "养了", "橘猫", "原文只有一只猫（叫代码），橘猫是品种；边无 chunk 属性"),
    ("代码", "住在", "小区楼下", "原文是「从小区楼下捡回来」，不是住在；边无 chunk 属性"),
]

_kw = neo4j_kwargs()
d = GraphDatabase.driver(_kw["url"], auth=(_kw["username"], _kw["password"]))
s = d.session()

print("模式:", "APPLY（删边）" if APPLY else "DRY-RUN（只读）")
print()

found = []
for src, rel, dst, why in TARGETS:
    rows = s.run(
        "MATCH (a {id:$a})-[r:`%s`]->(b {id:$b}) RETURN properties(r) AS props" % rel,
        a=src, b=dst,
    ).data()
    print("【%s -[%s]-> %s】" % (src, rel, dst))
    print("  理由: %s" % why)
    if not rows:
        print("  -> 边不存在，跳过\n")
        continue
    for r in rows:
        print("  边属性: %s" % r["props"])
        found.append({"src": src, "rel": rel, "dst": dst, "props": r["props"]})
    if APPLY:
        s.run(
            "MATCH (a {id:$a})-[r:`%s`]->(b {id:$b}) DELETE r" % rel,
            a=src, b=dst,
        )
        left = s.run(
            "MATCH (a {id:$a})-[r:`%s`]->(b {id:$b}) RETURN count(r) AS c" % rel,
            a=src, b=dst,
        ).single()["c"]
        print("  -> 已删除，剩余 %d 条" % left)
    print()

print("=== 「小李的宠物」事实链（两跳模板）===")
for r in s.run(
    "MATCH (p:人物 {id:'小李'})-[:养了]->(pet) "
    "OPTIONAL MATCH (pet)-[:品种是|属于*0..1]->(b) "
    "OPTIONAL MATCH (b)-[:属于类别*0..1]->(cat) "
    "RETURN pet.id AS pet, collect(DISTINCT b.id) AS mid, collect(DISTINCT cat.id) AS cat"
):
    print("  小李养了 %s -> 品种 %s -> 类别 %s" % (r["pet"], r["mid"], r["cat"]))

print()
print("=== 全部 养了 边现状 ===")
for r in s.run("MATCH (o)-[:养了]->(p) RETURN o.id AS o, p.id AS p ORDER BY o"):
    print("  %s 养了 %s" % (r["o"], r["p"]))

if APPLY and found:
    with open(BACKUP, "w", encoding="utf-8") as f:
        json.dump(found, f, ensure_ascii=False, indent=1)
    print("\n备份已写入 %s（%d 条）" % (BACKUP, len(found)))
elif not APPLY and found:
    print("\n[DRY-RUN] 将删除 %d 条边；加 --apply 执行" % len(found))
d.close()
