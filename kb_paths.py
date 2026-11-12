#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""索引目录解析：默认走生产库，可用 `RAG_CHROMA_DIR` 切到实验索引。

为什么要单独拆一个模块
─────────────────────
向量库路径原来在 `rag_agent.py` 和 `bm25_retriever.py` 里**各写了一处字面量**。
切索引时只要漏改一处，就会出现「向量检索查 A 语料、BM25 查 B 语料」的混合结果：
不报错、不告警，只是答案莫名其妙变差——这类静默错配必须从结构上消灭，
所以两边都只认这一个函数。

容错的取舍
──────────
配错的路径（不存在 / 空目录 / 没有 `chroma.sqlite3`）**一律回落到默认生产库**，
并把回落原因返回给调用方打日志。

宁可继续跑旧索引，也不要起一个空库：空库的表现是"什么问题都检索不到，
全部拒答"，看起来像模型坏了，排查成本极高。

判定标准是目录里有 `chroma.sqlite3`——chroma 是第一次写入才建这个文件，
所以"目录存在但没这个文件"就等于"没数据"，按不可用处理。
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Tuple

DEFAULT_DIRNAME = "chroma_db"
ENV_VAR = "RAG_CHROMA_DIR"
MARKER = "chroma.sqlite3"


def looks_like_index(path: str) -> bool:
    """目录里有没有 chroma 的数据库文件（空目录不算可用索引）。"""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, MARKER))


def resolve(base_dir: str, env: Optional[Mapping[str, str]] = None) -> Tuple[str, str]:
    """返回 `(索引目录, 说明)`。

    说明是给日志用的空串或原因；调用方不该据此改变行为——**目录一定是可用的**，
    这是这个函数唯一的承诺。
    """
    env = os.environ if env is None else env
    default = os.path.join(base_dir, DEFAULT_DIRNAME)
    raw = str(env.get(ENV_VAR) or "").strip()

    if not raw:
        return default, ""

    path = raw if os.path.isabs(raw) else os.path.join(base_dir, raw)

    if not os.path.isdir(path):
        return default, f"{ENV_VAR}={raw} 指向的目录不存在，已回落到 {DEFAULT_DIRNAME}"
    if not looks_like_index(path):
        return default, (f"{ENV_VAR}={raw} 目录里没有 {MARKER}（空库或未建完），"
                         f"已回落到 {DEFAULT_DIRNAME}")

    return path, ""
