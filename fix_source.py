# -*- coding: utf-8 -*-
"""fix_source.py —— 给图谱节点补 source 属性（来源文件）。

为什么需要它
────────────
graph_retriever.py 生成【图谱检索结果】时会正则提取节点的 source 属性
（r"'source': '([^']+)'"）来标注引用来源。如果节点没有 source，
所有图谱回答的来源都会退化成"本地知识图谱"，用户看不到原文出自哪个文件。

匹配策略（三档，从严到宽，命中即停）
────────────────────────────────
    T1 exact  原样字面包含                      —— 最可信
    T2 ci     大小写不敏感                      —— 覆盖 Mcn机构 / MCN机构 这类
    T3 norm   NFKC + 小写 + 去空白 + 去标点      —— 覆盖 "AJ1 联名款" / "AJ1联名款"

三档都不命中的，说明该节点是建图时 LLM 归纳出来的**抽象概念**
（如「水彩画」「海报设计」「游戏时长」），知识库里没有这个词，
此时保持 source 为 NULL 才是诚实的做法，绝不强行就近匹配。
脚本会在结尾把它们全列出来，便于日后从建图侧根治。

用法
────
    python fix_source.py --dry-run     # 只看匹配情况，不写库
    python fix_source.py               # 真写
    python fix_source.py --tier 1      # 只跑到 T1（回退到旧行为做对比）

连接参数从环境变量读，默认值对齐 docker-compose 里的邻居服务名。

回滚
────
    MATCH (n) REMOVE n.source
"""

import argparse
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", BASE_DIR / "knowledge_base"))

# 连接参数统一从 settings.py 取（口令不在这类脚本里写第二遍）
from settings import neo4j_kwargs  # noqa: E402

_neo4j = neo4j_kwargs()
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD = _neo4j["url"], _neo4j["username"], _neo4j["password"]

# 空白 + 常见中英文标点；用于 T3 的宽松匹配
_PUNCT = re.compile(
    r"[\s\u3000·、，,。.；;：:!！?？\"'“”‘’（）()\[\]【】《》〈〉<>/\\|_\-—－~`*#@&+=]+"
)


def norm_key(s: str) -> str:
    """NFKC 归一（全角→半角）→ 小写 → 去空白 → 去标点。"""
    s = unicodedata.normalize("NFKC", s).lower()
    return _PUNCT.sub("", s)


def load_knowledge() -> dict:
    """读全部 txt/md，返回 {相对路径: 内容}。"""
    if not KNOWLEDGE_DIR.exists():
        print(f"❌ 找不到知识库目录：{KNOWLEDGE_DIR}", file=sys.stderr)
        sys.exit(1)

    files = {}
    for p in sorted(KNOWLEDGE_DIR.rglob("*")):
        if p.suffix.lower() not in (".txt", ".md"):
            continue
        try:
            files[str(p.relative_to(KNOWLEDGE_DIR))] = p.read_text(
                encoding="utf-8", errors="ignore"
            )
        except OSError as e:
            print(f"⚠️ 读取失败 {p}: {e}", file=sys.stderr)
    return files


def match_source(node_id: str, files: dict, lowered: dict, normed: dict, max_tier: int):
    """返回 (tier, [文件名...])；tier=0 表示三档都未命中。"""
    if not node_id:
        return 0, []

    # T1 原样字面
    hit = [f for f, c in files.items() if node_id in c]
    if hit:
        return 1, hit

    if max_tier < 2:
        return 0, []

    # T2 大小写不敏感
    low = node_id.lower()
    hit = [f for f, c in lowered.items() if low in c]
    if hit:
        return 2, hit

    if max_tier < 3:
        return 0, []

    # T3 归一化（去空白/标点、全角转半角）
    nk = norm_key(node_id)
    if nk:
        hit = [f for f, c in normed.items() if nk in c]
        if hit:
            return 3, hit

    return 0, []


def main() -> None:
    parser = argparse.ArgumentParser(description="为图谱节点回填 source（来源文件）")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写库")
    parser.add_argument(
        "--tier", type=int, default=3, choices=[1, 2, 3], help="匹配放宽到第几档（默认 3）"
    )
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    print(f"🔗 连接 Neo4j：{NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:  # noqa: BLE001
        print(f"❌ 连接失败：{e}", file=sys.stderr)
        sys.exit(1)

    files = load_knowledge()
    print(f"📚 加载 {len(files)} 个知识文件")
    # 预计算两份索引，避免 740 × 22 次重复转换
    lowered = {k: v.lower() for k, v in files.items()}
    normed = {k: norm_key(v) for k, v in files.items()}

    with driver.session() as session:
        rows = session.run("MATCH (n) WHERE n.id IS NOT NULL RETURN n.id AS id").data()

    node_ids = [str(r["id"]) for r in rows]
    print(f"🔍 发现 {len(node_ids)} 个带 id 的节点")

    updates, tier_counter, no_source = [], Counter(), []
    for node_id in node_ids:
        tier, matched = match_source(node_id, files, lowered, normed, args.tier)
        if matched:
            tier_counter[tier] += 1
            updates.append({"id": node_id, "source": ", ".join(matched), "n": len(matched)})
        else:
            no_source.append(node_id)

    hit_counter = len(updates)
    coverage = hit_counter / len(node_ids) * 100 if node_ids else 0
    print(f"🎯 命中 {hit_counter} / {len(node_ids)}（覆盖率 {coverage:.1f}%）")
    print(
        f"   T1 原样 {tier_counter[1]} 个 | "
        f"T2 大小写 {tier_counter[2]} 个 | "
        f"T3 归一化 {tier_counter[3]} 个"
    )

    multi = [u for u in updates if u["n"] > 1]
    print(f"   其中跨文件出现 {len(multi)} 个（这些节点的来源会是多个文件）")

    # 看看常见的来源文件分布，便于判断匹配是否合理
    counter = Counter()
    for u in updates:
        for f in u["source"].split(", "):
            counter[f] += 1
    print("   来源分布 TOP5：")
    for name, c in counter.most_common(5):
        print(f"     {name}: {c} 个节点")

    if no_source:
        print(f"\n⚪ 无来源 {len(no_source)} 个（建图时归纳出的抽象概念，保持 NULL）：")
        for nid in no_source:
            print(f"     - {nid}")

    if args.dry_run:
        print("\n（--dry-run，未写库）")
        driver.close()
        return

    written = 0
    bs = max(1, args.batch_size)
    with driver.session() as session:
        for i in range(0, len(updates), bs):
            batch = updates[i:i + bs]
            session.run(
                """
                UNWIND $updates AS row
                MATCH (n {id: row.id})
                SET n.source = row.source
                """,
                updates=batch,
            )
            written += len(batch)
            print(f"   ⏳ 已写 {written} / {len(updates)}")

    with driver.session() as session:
        cnt = session.run(
            "MATCH (n) WHERE n.source IS NOT NULL RETURN count(n) AS c"
        ).single()["c"]

    driver.close()
    print(f"\n✅ 完成：本次写入 {written} 个节点，当前带 source 的节点共 {cnt} 个")


if __name__ == "__main__":
    main()
