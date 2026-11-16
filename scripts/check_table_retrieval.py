# -*- coding: utf-8 -*-
"""证明表格块真的在库里、而且能被检索到。

为什么要 `where={"is_table": True}` 过滤
────────────────────────────────────────
表格和它所在法的正文块语义高度重叠（都提"车船税""税率"），普通 top-5 里
往往是正文条款排在前面。要验证"表格入库并被检索"这件事本身，就得把检索
限定在表格块上 —— 这不是放宽标准，是隔离变量。

同时也跑一遍**不带过滤**的检索，看表格能不能自然进 top-5。两个结果都要看：
过滤内命中说明索引里有这张表，不过滤也命中才说明它在真实问答里用得上。
"""
import sys

sys.path.insert(0, "/home/zh/py_code")

import chromadb
from chromadb.config import Settings

from embeddings_shared import get_embeddings

DIR = "/home/zh/py_code/chroma_db_law_pdf"
QUERIES = [
    "车船税 1.6升以上至2.0升的乘用车 每年税额多少",
    "车船税税目税额表 摩托车 每辆",
    "大气污染物 每污染当量 多少元",
    "印花税 借款合同 税率",
    "资源税 原油 税率",
    "耕地占用税 每平方米 平均税额",
    "诉讼费用 财产案件 20万元 交纳多少",
    "仲裁委员会 案件受理费 收费标准",
]

emb = get_embeddings()
client = chromadb.PersistentClient(path=DIR, settings=Settings(anonymized_telemetry=False))
col = max(client.list_collections(), key=lambda c: c.count())
print(f"集合 {col.name} 共 {col.count()} 块")

res = col.get(where={"is_table": True}, include=["metadatas"])
n_tbl = len(res["ids"])
print(f"其中表格块: {n_tbl} 个")
files = {}
for m in res["metadatas"] or []:
    files[m.get("source_file", "?")] = files.get(m.get("source_file", "?"), 0) + 1
for f, n in sorted(files.items(), key=lambda x: -x[1])[:12]:
    print(f"  {n:>3} 块  {f}")

for q in QUERIES:
    print(f"\n问: {q}")
    vec = [emb.embed_query(q)]
    for tag, where in (("表内", {"is_table": True}), ("全库", None)):
        kw = {"where": where} if where else {}
        r = col.query(query_embeddings=vec, n_results=3, **kw)
        for rank, (doc, meta, dist) in enumerate(
                zip(r["documents"][0], r["metadatas"][0], r["distances"][0]), 1):
            head = doc.replace("\n", " ")[:80]
            print(f"  [{tag}{rank}] {1 - dist:.3f} {meta.get('source_file', '?')[:26]} | {head}")
