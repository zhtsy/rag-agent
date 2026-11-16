# -*- coding: utf-8 -*-
"""只测「检索段」耗时，不跑大模型 —— 用来回答"一题 20 秒里有多少是检索"。

评测脚本记录的是**端到端**耗时（检索 + 生成 + HTTP 往返），一个数字没法判断
该优化谁。这里把检索这一半单独拎出来计时：不起 HTTP 服务、不写索引、不调 LLM，
只读 chroma 目录，所以可以在评测后端停着的时候跑。

用法（环境变量必须在 import rag_agent 之前设好）：
    cd /home/zh/py_code
    RAG_CHROMA_DIR=chroma_db_law RAG_EMBED_PROVIDER=ollama RAG_EMBED_MODEL=bge-m3 \
      python3 scripts/profile_retrieval.py L001 T001 D008 R02

不给 id 就跑默认几条（1 条走精确查找 + 3 条走完整混合检索）。
"""

import json
import os
import sys
import time

os.environ.setdefault("RAG_TRACE_LOG", "0")      # 别让 trace 打到 stdout 里混着

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obs                                        # noqa: E402
import rag_agent                                  # noqa: E402

CASE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_cases_law.json"
)
DEFAULT_IDS = ["L001", "T001", "D008", "R02"]


def load_cases(ids):
    cases = json.load(open(CASE_FILE, encoding="utf-8"))
    by_id = {c["id"]: c for c in cases}
    return [(i, by_id[i]["question"]) for i in ids if i in by_id]


def main(argv):
    ids = argv[1:] or DEFAULT_IDS
    cases = load_cases(ids)

    print(f"索引目录: {os.environ.get('RAG_CHROMA_DIR', '(默认 chroma_db)')}")
    print(f"嵌入模型: {os.environ.get('RAG_EMBED_PROVIDER')}/{os.environ.get('RAG_EMBED_MODEL')}")
    print(f"用例数: {len(cases)}\n")

    # 预热：第一次调用要加载 chroma + bge-reranker（约 1.2GB），这段不能算进
    # 单题耗时，否则第一条永远是"最慢的那条"。用一句和用例无关的话把模型点着。
    t_warm = time.perf_counter()
    try:
        rag_agent._search_knowledge_base_impl(query="预热用的无关问题")
    except Exception as exc:
        print(f"[预热] 失败（不影响后续计时）: {exc!r}")
    print(f"[预热] {time.perf_counter() - t_warm:.1f}s\n")

    rows = []
    for cid, q in cases:
        trace = obs.start(q)
        t0 = time.perf_counter()
        try:
            out = rag_agent._search_knowledge_base_impl(query=q)
        except Exception as exc:                  # 单条失败不该中断整轮
            print(f"{cid} 检索异常: {exc!r}")
            continue
        wall = time.perf_counter() - t0
        stages = dict(trace.stages)
        rows.append((cid, wall, stages))
        detail = "  ".join(f"{k}={v/1000:.1f}s" for k, v in sorted(stages.items()))
        print(f"{cid}  检索段 {wall:5.1f}s  [{detail}]  返回 {len(out)} 字符")
        print(f"      问题: {q[:48]}")

    if len(rows) > 1:
        print("\n=== 汇总 ===")
        avg = sum(r[1] for r in rows) / len(rows)
        print(f"检索段均值 {avg:.1f}s（不含 LLM 生成）")
        names = sorted({k for _, _, s in rows for k in s})
        for n in names:
            vals = [s.get(n, 0.0) / 1000 for _, _, s in rows]
            print(f"  {n:12s} 均值 {sum(vals)/len(vals):5.1f}s  最大 {max(vals):5.1f}s")


if __name__ == "__main__":
    main(sys.argv)
