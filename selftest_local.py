#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selftest_local.py —— 本地直跑（WSL）的全链路自检

为什么需要它
────────────
云端有 eval_gate / 38 题评测 / pytest 三道闸，本地迁移过来之后都不等价：
本地的 LLM 是 Qwen+DeepSeek 混合、embedding 是 Ollama nomic、Neo4j 是单机实例。
任何一环没接上，表现都是"启动很慢 / 答非所问 / 一直转圈"，而不是一条明确的报错。
这个脚本把"能不能跑"拆成 5 组可判定的检查，逐项 PASS / FAIL / SKIP。

用法
────
    python selftest_local.py          # 全量（会加载 reranker，约 30~60 秒）
    python selftest_local.py --fast   # 跳过重模型加载（只查环境/服务/资产）
    python selftest_local.py --e2e    # 额外测已启动的后端 :8000（需先起服务）

退出码：0 = 没有 FAIL；1 = 至少一项 FAIL。
SKIP 不算失败（例如后端没起时跳过 e2e），但会在汇总里单独计数。
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- 结果收集
_ROWS = []


def rec(status, name, detail=""):
    """记录一项检查结果。status: PASS / FAIL / SKIP"""
    _ROWS.append((status, name, str(detail)))
    return status == "PASS"


def _count(status):
    return sum(1 for r in _ROWS if r[0] == status)


# ---------------------------------------------------------------- 早期环境
def load_env_early():
    """在 import local_env 之前把 .env 灌进 os.environ。

    不能只靠 dotenv：local_env 读不到变量时会静默走默认值，
    结果是"脚本以为检查了本地模式，其实根本没开"。
    手写解析，不覆盖已存在的真实环境变量（命令行 export 优先）。
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return 0
    n = 0
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def _env(key, default=""):
    return (os.environ.get(key) or default).strip()


def _env_flag(key, default=False):
    raw = _env(key)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


# ================================================================ 1. 运行环境
def check_env(env_keys):
    print("\n[1] 运行环境")
    py = sys.version_info
    rec("PASS" if py >= (3, 10) else "FAIL", "Python 版本",
        f"{py.major}.{py.minor}.{py.micro}" + ("" if py >= (3, 10) else "（需要 >=3.10）"))

    rec("PASS" if env_keys else "FAIL", ".env 读取",
        f"{env_keys} 个键" if env_keys else "文件缺失或为空")

    # 本地模式三件套：缺一项就是"启动卡几十秒"或"线程打架变慢"
    if _env_flag("RAG_LOCAL_MODE"):
        rec("PASS", "RAG_LOCAL_MODE", "=1（本地预设已开启）")
        rec("PASS" if os.environ.get("OMP_NUM_THREADS") == "1" else "FAIL",
            "OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "未设置"))
        if _env_flag("RAG_HF_OFFLINE", True):
            off = os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            rec("PASS" if off else "FAIL", "HF 离线模式",
                "TRANSFORMERS_OFFLINE=1" if off else "未生效（离线机会卡在联网重试）")
    else:
        rec("SKIP", "RAG_LOCAL_MODE", "未开启（按云端行为跑）")

    rec("PASS", "LLM provider",
        f"{_env('RAG_LLM_PROVIDER', 'deepseek')}（生成模型 {_env('RAG_LOCAL_LLM_MODEL', '-')}）")
    emb = _env("RAG_EMBED_PROVIDER", "bge")
    rec("PASS" if emb else "FAIL", "Embedding provider",
        f"{emb} / {_env('RAG_EMBED_MODEL', '默认')}")

    rec("PASS" if _env("DEEPSEEK_API_KEY") else "FAIL", "DEEPSEEK_API_KEY",
        "已配置" if _env("DEEPSEEK_API_KEY") else "缺失（DeepSeek 那一路会 401）")

    probe = [
        ("chromadb", "向量库"), ("langchain_chroma", "Chroma 适配"),
        ("langchain_ollama", "本地 Qwen/nomic"), ("langchain_deepseek", "DeepSeek"),
        ("FlagEmbedding", "重排模型"), ("rank_bm25", "BM25"), ("jieba", "中文分词"),
        ("fastapi", "后端"), ("neo4j", "图数据库驱动"),
    ]
    missing = []
    for mod, why in probe:
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            missing.append(f"{mod}({why})")
    rec("PASS" if not missing else "FAIL", "关键依赖导入",
        f"{len(probe)} 个全部可用" if not missing else "缺: " + ", ".join(missing))


# ================================================================ 2. 依赖服务
def _local_opener():
    """访问本机服务专用的 opener：显式禁用代理。

    WSL 里常设 http_proxy 做翻墙代理，urllib 默认会**连 127.0.0.1 也走代理**，
    结果是 Ollama 明明起着却报 `HTTP 502 Bad Gateway`——排查方向直接被带偏。
    这个坑在本次迁移自检里真实踩到过，所以固定绕开。
    """
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http_json(url, timeout=5):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with _local_opener().open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "ignore"))


def check_services():
    print("\n[2] 依赖服务")

    # ---- Ollama
    host = (_env("OLLAMA_BASE_URL") or _env("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    try:
        tags = _http_json(f"{host}/api/tags", timeout=5)
        names = [m.get("name", "") for m in tags.get("models", [])]
        rec("PASS", "Ollama 可连接", f"{host}（{len(names)} 个模型）")
        need = []
        for m in (_env("RAG_EMBED_MODEL", "nomic-embed-text"),
                  _env("RAG_LOCAL_LLM_MODEL", "qwen2.5:7b")):
            # ollama 的 tag 形如 qwen2.5:7b / nomic-embed-text:latest，按名字前缀匹配
            if not any(n == m or n.split(":")[0] == m.split(":")[0] for n in names):
                need.append(m)
        rec("PASS" if not need else "FAIL", "所需模型在位",
            "全部命中" if not need else "缺: " + ", ".join(need) + "（ollama pull <模型>）")
    except Exception as e:  # noqa: BLE001
        rec("FAIL", "Ollama 可连接", f"{host} → {type(e).__name__}: {e}；先 `ollama serve`")

    # ---- Neo4j
    uri = _env("NEO4J_URI", "bolt://localhost:7687")
    pwd = _env("NEO4J_PASSWORD")
    if not pwd:
        rec("FAIL", "Neo4j 口令", "NEO4J_PASSWORD 未配置")
        return
    try:
        from neo4j import GraphDatabase

        drv = GraphDatabase.driver(uri, auth=(_env("NEO4J_USER", "neo4j"), pwd),
                                   connection_timeout=5)
        drv.verify_connectivity()
        rec("PASS", "Neo4j 可连接", uri)
        with drv.session() as s:
            try:
                n = s.run("MATCH (n:__Entity__) RETURN count(n) AS c").single()["c"]
                rec("PASS" if n else "FAIL", "图谱实体数", f"{n} 个 __Entity__ 节点")
            except Exception as e:  # noqa: BLE001
                rec("FAIL", "图谱实体数", f"查询失败: {e}")
            try:
                # 专测迁移后的品种层级（backfill_breed.py 的产出）
                c = s.run("MATCH ()-[r:属于类别]->() RETURN count(r) AS c").single()["c"]
                rec("PASS" if c else "FAIL", "品种层级 `属于类别`",
                    f"{c} 条边" if c else "0 条边（跑 python backfill_breed.py --apply）")
            except Exception as e:  # noqa: BLE001
                rec("FAIL", "品种层级 `属于类别`", f"查询失败: {e}")
        drv.close()
    except Exception as e:  # noqa: BLE001
        rec("FAIL", "Neo4j 可连接", f"{uri} → {type(e).__name__}: {e}")


# ================================================================ 3. 数据资产
def check_assets():
    print("\n[3] 数据资产")

    db = ROOT / "chroma_db" / "chroma.sqlite3"
    if not db.exists():
        rec("FAIL", "chroma 库", "chroma_db/chroma.sqlite3 不存在")
    else:
        try:
            import sqlite3

            def _counts():
                """读块数与带 entity 的块数。

                先按普通只读模式开；被锁（chroma 正在写，或在 WSL 的 9p 网络盘上
                不支持 fcntl 文件锁）就退到 immutable=1 —— 那个模式完全不加锁，
                读到的是"某一时刻的快照"，对统计块数足够。
                """
                last = None
                for uri in (f"file:{db}?mode=ro", f"file:{db}?immutable=1"):
                    try:
                        con = sqlite3.connect(uri, uri=True, timeout=5)
                        cur = con.cursor()
                        cur.execute("select count(*) from embeddings")
                        n = cur.fetchone()[0]
                        cur.execute("select count(*) from embedding_metadata where key='entity'")
                        m = cur.fetchone()[0]
                        con.close()
                        return n, m
                    except sqlite3.OperationalError as e:  # noqa: PERF203
                        last = e
                raise last  # type: ignore[misc]

            n, with_entity = _counts()
            rec("PASS" if n else "FAIL", "chroma 块数", f"{n} 块")
            # entity 元数据是 filter={"entity": ...} 的前提；缺了会让带实体问题退化成全库扫
            rec("PASS" if with_entity >= n * 0.9 else "FAIL", "块级 entity 元数据",
                f"{with_entity}/{n}")
        except Exception as e:  # noqa: BLE001
            rec("FAIL", "chroma 库读取", f"{type(e).__name__}: {e}")

    kb = ROOT / "knowledge_base"
    files = [p for p in kb.rglob("*") if p.is_file()] if kb.exists() else []
    rec("PASS" if len(files) >= 10 else "FAIL", "BM25 语料文件", f"{len(files)} 篇")

    rk = Path(_env("RAG_RERANKER_PATH", "/home/zh/models/bge-reranker-base"))
    if not rk.is_absolute():
        rk = ROOT / rk
    weight = (list(rk.glob("*.bin")) + list(rk.glob("*.safetensors"))) if rk.exists() else []
    rec("PASS" if weight else "FAIL", "重排模型目录",
        f"{rk.name}/{weight[0].name}" if weight else f"{rk} 下没有权重文件")

    key_file = ROOT / ".rag_api_key"
    try:
        key = key_file.read_text(encoding="utf-8").strip()
        rec("PASS" if len(key) >= 16 else "FAIL", "访问密钥文件",
            f"长度 {len(key)}" if len(key) >= 16 else "内容过短，疑似没生成")
    except Exception:  # noqa: BLE001
        rec("FAIL", "访问密钥文件", ".rag_api_key 不存在（后端鉴权会 503）")


# ================================================================ 4. 组件级
def check_components(fast=False):
    print("\n[4] 组件级")

    # ---- embedding：维度必须和库里的向量一致，否则静默打崩检索
    emb = None
    try:
        from embeddings_shared import get_embeddings

        t0 = time.time()
        emb = get_embeddings()
        dim = len(emb.embed_query("测试"))
        rec("PASS" if dim == 768 else "FAIL", "embedding 维度",
            f"{type(emb).__name__} → {dim} 维，{time.time() - t0:.2f}s"
            + ("" if dim == 768 else "（库内是 768，维度不符会静默错配）"))
    except Exception as e:  # noqa: BLE001
        # 缺包和连不上是两个完全不同的修法，别混成一句话
        hint = "（依赖没装全，先 pip install -r requirements.txt）" \
            if isinstance(e, ImportError) else "（Ollama 没起？先 ollama serve）"
        rec("FAIL", "embedding 可用", f"{type(e).__name__}: {e}{hint}")

    # ---- BM25：语料读得到、分得出词、能返回结果
    try:
        from bm25_retriever import build_bm25_retriever

        t0 = time.time()
        r = build_bm25_retriever()
        docs = r.invoke("小智喜欢什么")
        rec("PASS" if docs else "FAIL", "BM25 检索",
            f"{len(docs)} 条命中，{time.time() - t0:.2f}s")
    except Exception as e:  # noqa: BLE001
        rec("FAIL", "BM25 检索", f"{type(e).__name__}: {e}")

    # ---- 向量检索：这一条同时验证"库能打开 + 维度匹配 + 有召回"
    if emb is not None:
        try:
            from langchain_chroma import Chroma

            t0 = time.time()
            vs = Chroma(persist_directory=str(ROOT / "chroma_db"), embedding_function=emb)
            hits = vs.similarity_search_with_score("小智喜欢什么", k=3)
            top = hits[0][1] if hits else 0.0
            rec("PASS" if hits else "FAIL", "向量检索",
                f"{len(hits)} 条，top1 距离 {top:.4f}，{time.time() - t0:.2f}s")
        except Exception as e:  # noqa: BLE001
            rec("FAIL", "向量检索", f"{type(e).__name__}: {e}")

    # ---- reranker：本地最重的一环，单独计时（CPU 降频时它会翻倍）
    if fast:
        rec("SKIP", "重排模型加载", "--fast 跳过")
    else:
        try:
            from FlagEmbedding import FlagReranker

            path = _env("RAG_RERANKER_PATH", "/home/zh/models/bge-reranker-base")
            t0 = time.time()
            rr = FlagReranker(path, use_fp16=False)
            load_s = time.time() - t0

            t1 = time.time()
            scores = rr.compute_score(
                [["小智喜欢什么", "小智喜欢猫"], ["小智喜欢什么", "今天天气不错"]],
                normalize=True,
            )
            pair_s = time.time() - t1
            good, bad = float(scores[0]), float(scores[1])
            rec("PASS" if good > bad else "FAIL", "重排判别力",
                f"相关 {good:.4f} > 无关 {bad:.4f}；加载 {load_s:.1f}s，2 对 {pair_s:.2f}s")
        except Exception as e:  # noqa: BLE001
            rec("FAIL", "重排模型加载", f"{type(e).__name__}: {e}")

    # ---- 检索工具：走完整 search_knowledge_base（图谱 → 向量+BM25 → 重排 → 门槛）
    if fast:
        rec("SKIP", "检索工具端到端", "--fast 跳过")
    else:
        try:
            from rag_agent import search_knowledge_base

            # @tool 装饰后是 StructuredTool，真正干活的是 .func；
            # 老版本没有 .func 就退回直接调用，免得脚本因 langchain 版本差异误报。
            fn = getattr(search_knowledge_base, "func", None) or search_knowledge_base
            t0 = time.time()
            out = fn("小智喜欢什么", "")
            dt = time.time() - t0
            ok = bool(out) and "未在知识库中找到" not in out
            rec("PASS" if ok else "FAIL", "检索工具端到端",
                f"{dt:.2f}s，返回 {len(out)} 字符" + ("" if ok else "，落进了拒答分支"))
        except Exception as e:  # noqa: BLE001
            rec("FAIL", "检索工具端到端", f"{type(e).__name__}: {e}")


# ================================================================ 5. 端到端
def check_e2e():
    print("\n[5] 后端端到端")
    base = _env("RAG_E2E_BASE", "http://127.0.0.1:8000").rstrip("/")

    try:
        h = _http_json(f"{base}/health", timeout=5)
        rec("PASS", "/health", f"status={h.get('status')} inflight={h.get('inflight')}")
    except Exception as e:  # noqa: BLE001
        rec("FAIL", "/health", f"{base} → {type(e).__name__}；后端没起？uvicorn main1:app --port 8000")
        return

    try:
        key = (ROOT / ".rag_api_key").read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        key = ""

    # 鉴权：无密钥必须 401。fail-open 是这层最危险的回归，必须每次都验
    body = json.dumps({"message": "你好", "session_id": "selftest"}).encode("utf-8")
    req = urllib.request.Request(f"{base}/chat/stream", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with _local_opener().open(req, timeout=10) as resp:
            code = resp.status
    except urllib.error.HTTPError as he:
        code = he.code
    except Exception as e:  # noqa: BLE001
        code = f"{type(e).__name__}: {e}"
    rec("PASS" if code == 401 else "FAIL", "无密钥应 401", f"实际 {code}")

    # 带密钥问一句：收 SSE 文本帧 + 终帧里的 sources/confidence/status
    q = "小智喜欢什么"
    body = json.dumps({"message": q, "session_id": f"selftest-{int(time.time())}"}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/stream", data=body,
        headers={"Content-Type": "application/json", "X-API-Key": key},
    )
    chunks, final = [], None
    t0 = time.time()
    try:
        with _local_opener().open(req, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:  # noqa: BLE001
                    continue
                if "text" in obj:
                    chunks.append(obj["text"])
                if "status" in obj or "sources" in obj:
                    final = obj
        dt = time.time() - t0
        answer = "".join(chunks)

        rec("PASS" if answer else "FAIL", "SSE 生成",
            f"{dt:.1f}s / {len(answer)} 字（含首 token 等待）")
        if final is None:
            rec("FAIL", "SSE 收尾帧", "没收到含 sources/status 的终帧")
        else:
            rec("PASS", "答案字段",
                f"status={final.get('status')} confidence={final.get('confidence')} "
                f"sources={len(final.get('sources') or [])} "
                f"contexts={len(final.get('contexts') or [])}")
            print(f"      ↳ {answer.replace(chr(10), ' ')[:70]}")
    except Exception as e:  # noqa: BLE001
        rec("FAIL", "SSE 端到端", f"{type(e).__name__}: {e}")


# ================================================================ 入口
def main():
    ap = argparse.ArgumentParser(description="本地 RAG 系统自检")
    ap.add_argument("--fast", action="store_true", help="跳过重模型加载")
    ap.add_argument("--e2e", action="store_true", help="额外测已启动的后端 :8000")
    args = ap.parse_args()

    print("=" * 70)
    print("本地 RAG 系统自检（WSL 直跑口径）")
    print("=" * 70)

    n_keys = load_env_early()
    # local_env 必须在任何 torch/transformers 之前 import —— 这正是它存在的理由
    try:
        import local_env  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"! local_env 导入失败: {e}（OMP/离线预设不会生效）")

    check_env(n_keys)
    check_services()
    check_assets()
    check_components(fast=args.fast)
    if args.e2e:
        check_e2e()

    print("\n" + "=" * 70)
    for status, name, detail in _ROWS:
        line = f"  {status}  {name:<24} {detail}"
        print(line[:150])
    p, f, s = _count("PASS"), _count("FAIL"), _count("SKIP")
    print("-" * 70)
    print(f"  合计 {len(_ROWS)} 项：PASS {p} / FAIL {f} / SKIP {s}")
    if f:
        print("\n  先修这些：")
        for status, name, detail in _ROWS:
            if status == "FAIL":
                print(f"   - {name}: {detail[:110]}")
    print("=" * 70)
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())


