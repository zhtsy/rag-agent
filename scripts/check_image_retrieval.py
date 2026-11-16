# -*- coding: utf-8 -*-
"""证明图片块真的在库里、而且能被检索到。

为什么要带 `where={"ext": ".png"}` 过滤
───────────────────────────────────────
同一部法的 PDF 块内容更完整（整页 OCR 全量），rerank 时会盖过图片块，
所以普通检索看不到图片来源。要验证"图片入库并被检索"这件事本身，
就得把检索限定在图片块上 —— 这不是放宽标准，是隔离变量。

同时打印图片块的元数据，确认 kind=image、有宽高、source_file 是页码形态。
"""
import sys

sys.path.insert(0, "/home/zh/py_code")

import chromadb
from chromadb.config import Settings

from embeddings_shared import get_embeddings

DIR = "/home/zh/py_code/chroma_db_law_pdf"
QUERIES = [
    "捐赠财产的使用有什么要求",
    "公益性社会团体受赠的财产属于什么性质",
]

emb = get_embeddings()
client = chromadb.PersistentClient(path=DIR, settings=Settings(anonymized_telemetry=False))
# 建库用的是 langchain 的 Chroma 包装，集合名叫 `langchain`；
# 取"块数最多的那个"比写死名字稳，也免得 get_or_create 在线上库里凭空建个空集合。
col = max(client.list_collections(), key=lambda c: c.count())

all_png = col.get(where={"ext": ".png"}, include=["metadatas"])
print(f"库里图片块: {len(all_png['ids'])} 个")
for m in (all_png["metadatas"] or [])[:3]:
    print("  元数据样例:", {k: m.get(k) for k in
                          ("source_file", "kind", "width", "height", "ext", "article")})

for q in QUERIES:
    res = col.query(query_embeddings=[emb.embed_query(q)], n_results=3,
                    where={"ext": ".png"})
    print(f"\n问: {q}")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1 - dist
        print(f"  {sim:.3f} {meta.get('source_file')} | {doc[:50]!r}")
