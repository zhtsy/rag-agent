"""按 geo_hierarchy 的映射给图谱补 `城市 -[:属于]-> 省` 边。

用法
----
    python backfill_geo.py                 # 只打印计划，不写库（默认）
    python backfill_geo.py --apply         # 真的写
    python backfill_geo.py --audit         # 只看现有属于边的来源分布
    python backfill_geo.py --rollback      # 删掉所有补出来的边

三个安全设计
------------
1. **默认 dry-run。** 这个脚本改的是线上图谱，误跑一次会污染 `属于` 语义。
   想看计划必须显式说 --apply。

2. **MERGE 而不是 CREATE。** 重复执行不产生重复边。
   扬州那条本来就存在（`江苏扬州` 文本共现抽出来的），这里是 no-op，
   正好也验证了幂等。

3. **冲突不覆盖，只报告。** 如果某城市已经有一条指向"别的省"的 `属于` 边，
   说明图谱里已有（可能是错的）信息，脚本跳过并计入 conflict，
   由人判断——自动覆盖等于把错误数据静默改写，事后无从追溯。

回滚
----
所有补出来的边都带 `origin='geo_backfill'`，删掉它们即可回到补边前的状态：

    MATCH ()-[r:属于]->() WHERE r.origin = 'geo_backfill' DELETE r

`--rollback` 执行的就是这句。抽取来的边没有这个属性，一条都不会碰。
"""

import argparse
import json
import os
import sys

from neo4j import GraphDatabase

from geo_hierarchy import (
    BACKFILL_ORIGIN,
    PROVINCE_NAMES,
    audit_cypher,
    plan_edges,
)

# 补边报告。回滚时靠它知道补过哪些省节点（节点上不存标记，见下方说明）。
REPORT_PATH = "/root/logs/geo_backfill.json"


def _driver():
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, pw), connection_timeout=10)


def load_state(session):
    """一次读全：节点 id 集合、现有属于边的出边映射、边的总数。"""
    nodes = {r["id"] for r in session.run(
        "MATCH (n) WHERE n.id IS NOT NULL RETURN n.id AS id").data()}

    existing = {}
    for r in session.run(
            "MATCH (a)-[r:属于]->(b) RETURN a.id AS a, b.id AS b, "
            "coalesce(r.origin, 'extracted') AS origin").data():
        existing.setdefault(r["a"], []).append(r["b"])

    total_edges = session.run(
        "MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    total_nodes = session.run(
        "MATCH (n) RETURN count(n) AS c").single()["c"]
    return nodes, existing, total_nodes, total_edges


def show_audit(session):
    print("=== 现有 `属于` 边按来源分布 ===")
    for row in session.run(audit_cypher()).data():
        print(f"  {row['origin']:<14} {row['edges']:>4} 条   例: {row['sample']}")


def _drop_orphan_nodes(session) -> int:
    """删掉补边时新建、现在已无任何边的省节点。

    节点上刻意不存"我是补出来的"标记——任何自定义属性都会被
    `graph_retriever._collect_entities` 当成实体属性渲染进答案
    （踩过：答案里出现「浙江是 geo_backfill」）。
    所以清单存在报告文件里，删之前再确认节点确实没有任何边。
    """
    if not os.path.exists(REPORT_PATH):
        return 0
    try:
        with open(REPORT_PATH, encoding="utf-8") as f:
            ids = json.load(f).get("new_nodes") or []
    except (ValueError, OSError):
        return 0

    removed = 0
    for nid in ids:
        rec = session.run(
            "MATCH (p:地点 {id: $p}) WHERE NOT (p)-[]-() "
            "DELETE p RETURN count(*) AS c", p=nid).single()
        removed += rec["c"] if rec else 0
    return removed


def main(argv=None):
    ap = argparse.ArgumentParser(description="补 城市→省 的地理层级边")
    ap.add_argument("--apply", action="store_true", help="真的写库（默认 dry-run）")
    ap.add_argument("--audit", action="store_true", help="只看属于边的来源分布")
    ap.add_argument("--rollback", action="store_true", help="删除所有补出来的边")
    args = ap.parse_args(argv)

    drv = _driver()
    try:
        with drv.session() as s:
            if args.audit:
                show_audit(s)
                return 0

            if args.rollback:
                n = s.run("MATCH ()-[r:属于]->() WHERE r.origin = $o "
                          "RETURN count(r) AS c", o=BACKFILL_ORIGIN).single()["c"]
                if n == 0:
                    print("没有需要回滚的边（图上没有 origin=geo_backfill 的属于边）")
                    # 边可能已经删过、只剩节点，仍然尝试清理（幂等）
                    _drop_orphan_nodes(s)
                    return 0
                s.run("MATCH ()-[r:属于]->() WHERE r.origin = $o DELETE r",
                      o=BACKFILL_ORIGIN)
                print(f"已删除 {n} 条补出来的边")
                print(f"已删除 {_drop_orphan_nodes(s)} 个补出来的省节点")
                print("图谱回到补边前状态")
                return 0

            nodes, existing, n_nodes_before, n_edges_before = load_state(s)
            edges, missing_provinces = plan_edges(nodes)

            # 分成四类：可补、已存在、冲突、省节点需新建
            to_add, already, conflicts = [], [], []
            for e in edges:
                cur = existing.get(e.city) or []
                if e.province in cur:
                    already.append(e)
                elif any(t in PROVINCE_NAMES for t in cur):
                    # 已经指向另一个省 → 数据冲突，交给人
                    conflicts.append((e, [t for t in cur if t in PROVINCE_NAMES]))
                else:
                    to_add.append(e)

            print(f"图谱现状：{n_nodes_before} 节点 / {n_edges_before} 边")
            print(f"计划补边 {len(edges)} 条："
                  f"新增 {len(to_add)}，已存在 {len(already)}，冲突 {len(conflicts)}")
            if missing_provinces:
                print(f"需新建省节点 {len(missing_provinces)} 个：{missing_provinces}")
            print()
            for e in to_add:
                print(f"  + ({e.city})-[:属于]->({e.province})")
            for e in already:
                print(f"  = ({e.city})-[:属于]->({e.province})  已存在，跳过")
            for e, cur in conflicts:
                print(f"  ! ({e.city}) 已指向 {cur}，与映射 {e.province} 冲突，跳过")

            if not args.apply:
                print("\n[dry-run] 未写库。确认无误后加 --apply 执行。")
                return 0

            # ---- 写：先建缺失的省节点，再建边 ----
            created_nodes = 0
            new_node_ids: list = []
            for prov in missing_provinces:
                # 显式查一次再建：用 MERGE 的 ON CREATE 反推"是否新建"在
                # 重复执行时会误判（节点已存在但带着上次写的 created_by）。
                # 建节点会改变节点总数，这个数字必须准确，不能靠推断。
                existed = s.run("MATCH (p {id: $p}) RETURN count(p) AS c",
                                p=prov).single()["c"] > 0
                if existed:
                    continue
                # 格式对齐 build_graph.write_graph：先 MERGE __Entity__ 再 SET 标签
                #
                # 这里只设标签和 id，**刻意不加任何自定义属性**。
                # 两个都踩过：
                #   source    —— 会被 _collect_entities 当引用来源展示，等于给一个假出处
                #   created_by —— 会进实体的 attrs，被渲染成「浙江（geo_backfill）」，
                #                 模型看到就把内部标记写进答案了
                # 节点层面的审计不需要额外字段：补出来的边带 origin 标记，
                # 顺着边就能找到这些节点（见 audit_cypher 的 geod 变体）。
                s.run(
                    "MERGE (p:`__Entity__` {id: $p}) SET p:`地点`", p=prov)
                created_nodes += 1
                new_node_ids.append(prov)

            created_edges = 0
            for e in to_add:
                rec = s.run(
                    "MATCH (c {id: $c}) MATCH (p {id: $p}) "
                    "MERGE (c)-[r:属于]->(p) "
                    "ON CREATE SET r.origin = $o, r.geo = true "
                    "RETURN r.origin AS origin",
                    c=e.city, p=e.province, o=BACKFILL_ORIGIN).single()
                if rec and rec["origin"] == BACKFILL_ORIGIN:
                    created_edges += 1

            _, _, n_nodes_after, n_edges_after = load_state(s)
            print()
            print(f"已写入：新建省节点 {created_nodes} 个，新增边 {created_edges} 条")
            print(f"图谱现状：{n_nodes_after} 节点 / {n_edges_after} 边 "
                  f"（变动 {n_nodes_after - n_nodes_before:+d} / "
                  f"{n_edges_after - n_edges_before:+d}）")
            print()
            show_audit(s)
            print(f"\n回滚：python backfill_geo.py --rollback   "
                  f"（或 MATCH ()-[r:属于]->() WHERE r.origin='{BACKFILL_ORIGIN}' DELETE r）")

            # 落一份结果，便于写进文档时引用
            report = {
                "nodes_before": n_nodes_before, "nodes_after": n_nodes_after,
                "edges_before": n_edges_before, "edges_after": n_edges_after,
                "created_nodes": created_nodes, "created_edges": created_edges,
                "new_nodes": new_node_ids,
                "planned": len(edges), "already": len(already),
                "conflicts": [{"city": e.city, "want": e.province, "has": cur}
                              for e, cur in conflicts],
                "added": [{"city": e.city, "province": e.province,
                           "self_ref": e.is_self_ref} for e in to_add],
            }
            out = REPORT_PATH
            os.makedirs(os.path.dirname(out), exist_ok=True)
            # 与历史报告合并 new_nodes：第二次跑 --apply 时 created_nodes=0，
            # 若直接覆盖，回滚就不知道第一次建过哪些节点了。
            prev_nodes = []
            if os.path.exists(out):
                try:
                    with open(out, encoding="utf-8") as f:
                        prev_nodes = json.load(f).get("new_nodes") or []
                except (ValueError, OSError):
                    prev_nodes = []
            report["new_nodes"] = sorted(set(prev_nodes) | set(new_node_ids))
            with open(out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"报告已写入 {out}")
            return 0
    finally:
        drv.close()


if __name__ == "__main__":
    sys.exit(main())
