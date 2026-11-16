# -*- coding: utf-8 -*-
"""端到端 profile 一题：检索 / 首字 / 生成各占多少。

`profile_retrieval.py` 只测检索段，这个脚本走和线上评测**完全相同**的入口
（`stream_conversation` → SSE 队列），因此能回答"20 秒里检索占几秒、生成占几秒"。

不起 HTTP 服务、不写索引，直接调协程。评测后端停着也能跑。

用法：
    cd /home/zh/py_code
    RAG_CHROMA_DIR=chroma_db_law RAG_EMBED_PROVIDER=ollama RAG_EMBED_MODEL=bge-m3 \
      python3 scripts/profile_answer.py T001 D008
"""

import asyncio
import json
import os
import sys
import time

os.environ.setdefault("RAG_TRACE_LOG", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obs                                        # noqa: E402
import rag_agent                                  # noqa: E402

CASE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_cases_law.json"
)
DEFAULT_IDS = ["T001"]


async def ask(cid, q, round_no):
    # thread_id 带时间戳：同一个 id 反复测时必须换会话，否则 MySQL 里上一轮的历史
    # 会被 _seed_messages 接回来，实测耗时就混进了"历史越长越慢"这个变量。
    thread_id = f"prof_{cid}_{round_no}_{int(time.time())}"
    trace = obs.start(q)
    t0 = time.perf_counter()
    ttft = None
    pieces = []
    async for chunk in rag_agent.stream_conversation(q, thread_id=thread_id):
        # stream_conversation 吐的是 SSE 行（data: {...}），这里只关心"第一个字何时到"
        if ttft is None and chunk:
            ttft = time.perf_counter() - t0
        pieces.append(chunk)
    total = time.perf_counter() - t0
    return total, ttft, trace.stages, "".join(pieces)


async def main(argv):
    ids = argv[1:] or DEFAULT_IDS
    cases = {c["id"]: c["question"] for c in json.load(open(CASE_FILE, encoding="utf-8"))}

    print(f"索引: {os.environ.get('RAG_CHROMA_DIR')} ｜ 生成模型: "
          f"{os.environ.get('RAG_LLM_PROVIDER')}/qwen2.5:7b\n")

    # 预热一次就够（加载 chroma + reranker + 生成模型）。**必须用无关问题**：
    # 拿同一道题预热会让检索缓存命中，正式那次跳过 embedding 直接进生成，
    # 测出来偏乐观，恰好会把"每题重载模型"这个真问题掩盖掉。
    t_warm = time.perf_counter()
    await ask("warm", "预热用的无关问题，请随便回一句", "warm")
    print(f"[预热] {time.perf_counter() - t_warm:.1f}s（含模型加载，不计入统计）\n")

    rows = []
    for cid in ids:
        q = cases[cid]
        total, ttft, stages, out = await ask(cid, q, "1")
        rows.append((cid, total, ttft, stages, len(out)))
        detail = "  ".join(f"{k}={v/1000:.1f}s" for k, v in sorted(stages.items()))
        print(f"{cid}  总 {total:5.1f}s ｜ 首字 {ttft:5.1f}s ｜ 生成 {total-(ttft or 0):5.1f}s ｜ "
              f"SSE {len(out)} 字符")
        print(f"      stages: {detail or '(无)'}")
        print(f"      问题: {q[:50]}\n")

    if len(rows) > 1:
        print("=== 汇总 ===")
        avg = sum(r[1] for r in rows) / len(rows)
        avg_ttft = sum(r[2] or 0 for r in rows) / len(rows)
        print(f"端到端均值 {avg:.1f}s，其中首字前 {avg_ttft:.1f}s，首字后 {avg-avg_ttft:.1f}s")


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
