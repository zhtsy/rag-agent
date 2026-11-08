# -*- coding: utf-8 -*-
"""embeddings_shared.py —— 全项目共享的向量化模型入口

两种 provider，由 `RAG_EMBED_PROVIDER` 选择：

| provider | 实现 | 用在哪 |
|---|---|---|
| `bge`（**默认**） | sentence-transformers 加载 BAAI/bge-base-zh-v1.5 | 云端容器 |
| `ollama` | Ollama 的 nomic-embed-text | 本地 WSL（既有 chroma_db 就是它建的） |

默认值刻意保持 `bge`：云端不设这个变量，行为与迁移前完全一致。

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
# 本地 Ollama 的向量模型；与 py_code/chroma_db 里已有的向量保持一致
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"
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
        return OllamaEmbeddings(model=model, base_url=host)

    return LocalEmbeddings((os.getenv("RAG_EMBED_MODEL") or DEFAULT_MODEL).strip())

