# -*- coding: utf-8 -*-
"""embeddings_shared.py —— 全项目共享的向量化模型入口

两种 provider，由 `RAG_EMBED_PROVIDER` 选择：

| provider | 实现 | 用在哪 |
|---|---|---|
| `bge`（**默认**） | sentence-transformers 加载 BAAI/bge-base-zh-v1.5 | 云端容器 |
| `ollama` | Ollama 的 bge-m3（中文语料建库用它） | 本地 WSL，主口径 `chroma_db_law_pdf` 就是它建的 |

默认值刻意保持 `bge`：云端不设这个变量，行为与迁移前完全一致。
（`ollama` 分支的默认值 2026-11-16 从 `nomic-embed-text` 改成 `bge-m3`——
英文模型建中文法律库会让排序反转，见 `index_meta.py` 开头的事故记录。）

⚠️ **换 provider 等于换向量空间，必须重建 chroma_db。**
两个模型输出都是 768 维，chroma 不会报任何错，只会静默把检索质量打崩
（query 和库内向量不在同一分布）。这是本项目最容易踩且最难发现的一种失配。

背景：原来三处各自 new 一个 SentenceTransformer：
  - rag_agent.py::LocalEmbeddings
  - bm25_retriever.py::build_bm25_retriever 内部的 _LocalEmb
  - build_knowledge_base.py::LocalEmbeddings
其中前两者在 backend 的同一个 uvicorn 进程里，等于把同一个约 400MB 的模型
在内存里装了两份。这里收敛成按模型名缓存、线程安全的单例。

行为约定（务必保持，否则会静默降低检索质量）：
1. **不给 query 加 bge 的 instruction 前缀**。现有向量库是按裸文本建的，
   一旦给 query 加前缀，query 向量与库内向量就不在同一分布上，检索会崩。
2. 依旧 device="cpu"、normalize_embeddings=True，与原实现逐字一致。
"""

import os
import threading
from pathlib import Path
from typing import Dict

from langchain_core.embeddings import Embeddings

# 命令行脚本（build_knowledge_base / backfill_* / eval）不经过 settings.py，
# 也就没人加载 .env —— 于是 get_embeddings() 读不到 RAG_EMBED_PROVIDER，
# 静默退回 bge，把库建成另一个向量空间。这里兜底读一次项目根目录的 .env。
_ENV_FILE = Path(__file__).resolve().with_name(".env")
if _ENV_FILE.is_file() and not os.getenv("RAG_EMBED_PROVIDER"):
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_FILE)
    except ImportError:  # 云端镜像里没装 python-dotenv 时退化为纯环境变量
        pass

DEFAULT_MODEL = "BAAI/bge-base-zh-v1.5"
# 本地 Ollama 的向量模型；与主口径 chroma_db_law_pdf（bge-m3）保持一致。
# 早期 demo 库 chroma_db 是 nomic-embed-text 建的，切那套要显式设 RAG_EMBED_MODEL。
DEFAULT_OLLAMA_MODEL = "bge-m3"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# model_name -> SentenceTransformer 实例
_encoders: Dict[str, object] = {}
_lock = threading.Lock()


def get_encoder(model_name: str = DEFAULT_MODEL):
    """按模型名取编码器；同一进程内重复调用返回同一个对象，不重复占内存。"""
    enc = _encoders.get(model_name)
    if enc is not None:
        return enc
    with _lock:
        enc = _encoders.get(model_name)
        if enc is None:
            # 延迟导入，让 import 本模块保持轻量
            from sentence_transformers import SentenceTransformer

            enc = SentenceTransformer(model_name, device="cpu")
            _encoders[model_name] = enc
    return enc


class LocalEmbeddings(Embeddings):
    """LangChain Embeddings 适配层，模型本体来自共享单例。"""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = get_encoder(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def get_embeddings() -> Embeddings:
    """按 `RAG_EMBED_PROVIDER` 返回 embedding 实现。

    - 不设 / `bge`：走 LocalEmbeddings（sentence-transformers），保持云端行为不变
    - `ollama`：走 OllamaEmbeddings，与本地既有 chroma_db 的向量空间一致

    调用方（rag_agent / bm25_retriever / build_knowledge_base）统一用这个函数，
    不要在别处直接 new LocalEmbeddings —— 否则一换 provider 就会漏改某处，
    表现为"一部分路径用 bge、一部分用 ollama"，检索结果随机变差。
    """
    provider = (os.getenv("RAG_EMBED_PROVIDER") or "bge").strip().lower()

    if provider == "ollama":
        # 延迟导入：云端镜像即使没装 langchain-ollama 也不影响 bge 路径
        from langchain_ollama import OllamaEmbeddings

        host = (
            os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or DEFAULT_OLLAMA_HOST
        ).strip()
        model = (os.getenv("RAG_EMBED_MODEL") or DEFAULT_OLLAMA_MODEL).strip()

        # 嵌入模型强制跑 CPU，**不要**交回 ollama 自动调度。
        #
        # 本机是 6 GB 显存的 3060 Laptop：生成用 qwen2.5:7b 要 4.9 GiB，bge-m3 要 1.6 GiB，
        # 合计 6.5 GiB —— 装不下。于是每道题都在"检索加载嵌入模型 → 踢掉生成模型 →
        # 生成再踢回来"之间来回换，每次换出还要等显存回收约 5 秒。实测 47 道题触发
        # 57 次 runner 重载、168 次 VRAM 回收等待，单题凭空多出 15 秒左右。
        #
        # bge-m3 只有 566 M 参数，一次 query 的 embedding 在 CPU 上是几十毫秒，
        # 让出这 1.6 GiB 远比来回重载便宜。注意 **只是算力位置变了，权重和向量空间没变**，
        # 所以不需要重建索引（换 provider / 换模型才需要）。
        # 想回到"自动调度"就把 RAG_EMBED_NUM_GPU 设成 -1。
        try:
            _num_gpu = int(os.getenv("RAG_EMBED_NUM_GPU") or 0)
        except (TypeError, ValueError):
            _num_gpu = 0
        return OllamaEmbeddings(model=model, base_url=host, num_gpu=_num_gpu)

    return LocalEmbeddings((os.getenv("RAG_EMBED_MODEL") or DEFAULT_MODEL).strip())

