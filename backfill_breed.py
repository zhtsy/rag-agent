# -*- coding: utf-8 -*-
"""按 breed_hierarchy 给图谱补 `品种 -[:属于类别]-> 类别` 边，并清掉重复的 type 属性。

用法
----
    python backfill_breed.py                  # 只打印计划，不写库（默认）
    python backfill_breed.py --apply          # 真的写
    python backfill_breed.py --audit          # 只看 属于类别 边的来源分布
    python backfill_breed.py --rollback       # 删掉补出来的边与新建的类别节点

它解决两个问题
--------------
1. **聚合问法少人。** `(宠物)-[:品种是]->(品种)` 只到品种层，
   `WHERE 品种="猫"` 查不到"橘猫"，于是"养猫的是谁"会漏人。
   补上 `橘猫 -[:属于类别]-> 猫` 之后，`-[:属于类别*0..1]->` 一条路径就能覆盖
   "品种本身就是类别"和"品种属于某类别"两种情况。

2. **`type` 属性的假信号。** 宠物里只有"豆豆"带 `type: 金毛`，其余都走 `品种是`。
   生成 Cypher 的模型看到 Schema 里有 `type`，就会写成 `WHERE pet.type = "猫"`，
   而这条属性只存在于一个节点上，命中率为 0。用 `--clean-type` 删掉它
   （品种信息在 `品种是` 边里已经有一份，删的是纯重复）。

三个安全设计（与 backfill_geo 一致）
------------------------------------
1. **默认 dry-run。** 改的是线上图谱，想看计划必须显式 --apply。
2. **MERGE 而不是 CREATE。** 重复执行不产生重复边。
3. **冲突不覆盖，只报告。**

回滚
----
补出来的边带 `origin='breed_backfill'`；新建的类别节点（如 `狗`）只带标签和 id，
清单落在报告文件里（**不往节点上写任何自定义属性**——踩过：答案里会冒出
「浙江是 geo_backfill」）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from breed_hierarchy import (
    BREED_BACKFILL_ORIGIN,
    CATEGORY_LABEL,
    CATEGORY_REL,
    audit_cypher,
    plan_edges,
)

# 必须显式加载：不加载就会用下面的默认值 bolt://neo4j:7687（docker-compose 的服务名），
# 在宿主机上直接 DNS 解析失败。云端靠 compose 注入环境变量，本来不需要这行；
# 加了也只会"有 .env 时用 .env、没有时用注入值"，两边都对。
load_dotenv(Path(__file__).with_name(".env"))

def _pick_log_dir() -> Path:
    """云端优先 /root/logs（便于采集），本地落到项目内。

    这里不能用 `is_dir()` 直接判：`/root/logs` 是存在的，但 zh 用户无权限，
    `stat()` 抛的是 PermissionError 而不是返回 False——直接判会把脚本跑挂。
    """
    cand = Path(os.getenv("RAG_LOG_DIR") or "/root/logs")
    try:
        if cand.is_dir() and os.access(cand, os.W_OK):
            return cand
    except OSError:
        pass
    fallback = Path(__file__).resolve().parent / "logs"
    fallback.mkdir(exist_ok=True)
    return fallback


REPORT_PATH = str(_pick_log_dir() / "breed_backfill.json")

# 宠物上那个"只有一个人有"的品种属性。只清这一个键，不动别的。
STALE_PROP = "type"


def _driver():
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, pw), connection_timeout=10)


def load_state(session):
    nodes = {r["id"] for r in session.run(
        "MATCH (n) WHERE n.id IS NOT NULL RETURN n.id AS id").data()}
    existing = {}
    for r in session.run(
            "MATCH (a)-[r]->(b) WHERE type(r) = $rel RETURN a.id AS a, b.id AS b, "
            "coalesce(r.origin, 'extracted') AS origin", rel=CATEGORY_REL).data():
        existing.setdefault(r["a"], []).append(r["b"])
    total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    total_edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return nodes, existing, total_nodes, total_edges


def show_stale_prop(session):
    """列出仍带 type 属性的宠物节点（已经与 品种是 重复的那些）。"""
    rows = session.run(
        f"MATCH (n) WHERE n.{STALE_PROP} IS NOT NULL "
        f"RETURN n.id AS id, n.{STALE_PROP} AS val, labels(n) AS l ORDER BY id").data()
    return rows


def show_audit(session):
    print(f"=== 现有 `{CATEGORY_REL}` 边按来源分布 ===")
    rows = session.run(audit_cypher()).data()
    if not rows:
        print("  （空：还没有任何层级边）")
    for row in rows:
        print(f"  {row['origin']:<14} {row['edges']:>4} 条   例: {row['sample']}")


def _drop_orphan_nodes(session) -> int:
    """删掉补边时新建、现在已无任何边的类别节点。

    节点上刻意不存"我是补出来的"标记：任何自定义属性都会被
    `graph_retriever._collect_entities` 当成实体属性渲染进答案。
    清单存在报告文件里，删之前再确认节点确实没有任何边。
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
            f"MATCH (p:{CATEGORY_LABEL} {{id: $p}}) WHERE NOT (p)-[]-() "
            "DELETE p RETURN count(*) AS c", p=nid).single()
        removed += rec["c"] if rec else 0
    return removed


def main(argv=None):
    ap = argparse.ArgumentParser(description="补 品种→类别 的层级边")
    ap.add_argument("--apply", action="store_true", help="真的写库（默认 dry-run）")
    ap.add_argument("--audit", action="store_true", help=f"只看 {CATEGORY_REL} 边的来源分布")
    ap.add_argument("--rollback", action="store_true", help="删除补出来的边与新建节点")
    ap.add_argument("--clean-type", action="store_true",
                    help=f"同时清理宠物节点上重复的 `{STALE_PROP}` 属性")
    args = ap.parse_args(argv)

    drv = _driver()
    try:
        with drv.session() as s:
            if args.audit:
                show_audit(s)
                return 0

            if args.rollback:
                n = s.run("MATCH ()-[r]->() WHERE type(r) = $rel AND r.origin = $o "
                          "RETURN count(r) AS c",
                          rel=CATEGORY_REL, o=BREED_BACKFILL_ORIGIN).single()["c"]
                if n:
                    s.run("MATCH ()-[r]->() WHERE type(r) = $rel AND r.origin = $o DELETE r",
                          rel=CATEGORY_REL, o=BREED_BACKFILL_ORIGIN)
                    print(f"已删除 {n} 条补出来的边")
                else:
                    print("没有需要回滚的边")
                print(f"已删除 {_drop_orphan_nodes(s)} 个补出来的类别节点")

                # 恢复被清掉的 type 属性
                if os.path.exists(REPORT_PATH):
                    try:
                        with open(REPORT_PATH, encoding="utf-8") as f:
                            removed = json.load(f).get("removed_props") or []
                    except (ValueError, OSError):
                        removed = []
                    for item in removed:
                        s.run(f"MATCH (n {{id: $i}}) SET n.{STALE_PROP} = $v",
                              i=item["id"], v=item["value"])
                    if removed:
                        print(f"已恢复 {len(removed)} 个被清掉的 `{STALE_PROP}` 属性")
                print("图谱回到补边前状态")
                return 0

            nodes, existing, n_nodes_before, n_edges_before = load_state(s)
            edges, missing_categories = plan_edges(nodes)

            to_add, already, conflicts = [], [], []
            for e in edges:
                cur = existing.get(e.breed) or []
                if e.category in cur:
                    already.append(e)
                elif cur:
                    conflicts.append((e, cur))
                else:
                    to_add.append(e)

            stale = show_stale_prop(s)

            print(f"图谱现状：{n_nodes_before} 节点 / {n_edges_before} 边")
            print(f"计划补边 {len(edges)} 条：新增 {len(to_add)}，已存在 {len(already)}，"
                  f"冲突 {len(conflicts)}")
            if missing_categories:
                print(f"需新建类别节点 {len(missing_categories)} 个：{missing_categories}")
            print()
            for e in to_add:
                print(f"  + {e}")
            for e in already:
                print(f"  = {e}  已存在，跳过")
            for e, cur in conflicts:
                print(f"  ! ({e.breed}) 已指向 {cur}，与映射 {e.category} 冲突，跳过")
            if stale:
                print(f"\n`{STALE_PROP}` 属性（与 品种是 重复）:")
                for r in stale:
                    print(f"  - ({r['id']}).{STALE_PROP} = {r['val']!r}  labels={r['l']}")
                if not args.clean_type:
                    print("  （要清掉加 --clean-type）")

            if not args.apply:
                print("\n[dry-run] 未写库。确认无误后加 --apply 执行。")
                return 0

            # ---- 写：先建缺失的类别节点，再建边 ----
            created_nodes = 0
            new_node_ids = []
            for cat in missing_categories:
                existed = s.run("MATCH (p {id: $p}) RETURN count(p) AS c",
                                p=cat).single()["c"] > 0
                if existed:
                    # 节点已存在但可能没标签，补一个，方便 Cypher 写 (cat:类别)
                    s.run(f"MATCH (p {{id: $p}}) SET p:`{CATEGORY_LABEL}`", p=cat)
                    continue
                # 只设标签和 id，不加任何自定义属性（踩过 source/created_by 两个坑）
                s.run(f"MERGE (p:`__Entity__` {{id: $p}}) SET p:`{CATEGORY_LABEL}`", p=cat)
                created_nodes += 1
                new_node_ids.append(cat)

            # 已存在的类别节点也补标签，让 `(cat:类别)` 这种写法可用
            for e in to_add + already:
                s.run(f"MATCH (p {{id: $p}}) SET p:`{CATEGORY_LABEL}`", p=e.category)

            created_edges = 0
            for e in to_add:
                rec = s.run(
                    "MATCH (b {id: $b}) MATCH (c {id: $c}) "
                    f"MERGE (b)-[r:{CATEGORY_REL}]->(c) "
                    "ON CREATE SET r.origin = $o, r.breed = true "
                    "RETURN r.origin AS origin",
                    b=e.breed, c=e.category, o=BREED_BACKFILL_ORIGIN).single()
                if rec and rec["origin"] == BREED_BACKFILL_ORIGIN:
                    created_edges += 1

            # ---- 清理重复属性 ----
            removed_props = []
            if args.clean_type and stale:
                for r in stale:
                    removed_props.append({"id": r["id"], "value": r["val"]})
                    s.run(f"MATCH (n {{id: $i}}) REMOVE n.{STALE_PROP}", i=r["id"])

            _, _, n_nodes_after, n_edges_after = load_state(s)
            print()
            print(f"已写入：新建类别节点 {created_nodes} 个，新增边 {created_edges} 条，"
                  f"清理 `{STALE_PROP}` 属性 {len(removed_props)} 个")
            print(f"图谱现状：{n_nodes_after} 节点 / {n_edges_after} 边 "
                  f"（变动 {n_nodes_after - n_nodes_before:+d} / "
                  f"{n_edges_after - n_edges_before:+d}）")
            print()
            show_audit(s)
            print(f"\n回滚：python backfill_breed.py --rollback")

            report = {
                "nodes_before": n_nodes_before, "nodes_after": n_nodes_after,
                "edges_before": n_edges_before, "edges_after": n_edges_after,
                "created_nodes": created_nodes, "created_edges": created_edges,
                "removed_props": removed_props,
                "planned": len(edges), "already": len(already),
                "conflicts": [{"breed": e.breed, "want": e.category, "has": cur}
                              for e, cur in conflicts],
                "added": [{"breed": e.breed, "category": e.category} for e in to_add],
            }
            out = REPORT_PATH
            os.makedirs(os.path.dirname(out), exist_ok=True)
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
