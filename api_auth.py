# -*- coding: utf-8 -*-
"""api_auth.py —— 后端访问密钥的**唯一**读取点（后端 / 前端 / 评测脚本共用）。

为什么单独抽一个模块
────────────────────
密钥原本没有，谁都打得到 `/chat/stream`，等于把 DeepSeek 额度挂在公网上。
加鉴权后，"谁是可信调用方"这件事必须只有一个判定口径，否则前后端各读各的、
某天只改了一处，就会出现"前端 401、评测 200"这种难查的错位。

密钥从哪来（按优先级）
──────────────────────
1. 环境变量 `RAG_API_KEY`（容器/CI 里注入用）
2. 密钥文件，依次找：
     /app/.rag_api_key                  ← 容器内（项目目录 bind mount 到 /app）
     <本文件所在目录>/.rag_api_key       ← 宿主机直接跑脚本时
     /root/rag_deploy/.rag_api_key      ← 宿主机兜底

为什么用文件而不是写进 .env
──────────────────────────
改 `.env` 要 `docker compose up -d` **重建容器**（env 变更 restart 不生效），
而密钥文件是 bind mount，改完**下一次请求就生效**，不用重启、不用停机。
这也意味着轮换密钥时：改文件 → 立刻生效，旧密钥立刻失效。

用法
────
    from api_auth import api_key, auth_headers
    headers = {"Content-Type": "application/json", **auth_headers()}
"""

import os
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent

_KEY_FILE_CANDIDATES = (
    Path("/app/.rag_api_key"),
    _APP_DIR / ".rag_api_key",
    Path("/root/rag_deploy/.rag_api_key"),
)


def api_key() -> str:
    """返回当前生效的访问密钥；没有配置则返回空串。

    每次都重新读文件（不缓存）：改完密钥文件立即生效，删掉也能立即失效。
    单次 read_text 只有几微秒，对本项目的 QPS 完全不构成负担。
    """
    env = (os.environ.get("RAG_API_KEY") or "").strip()
    if env:
        return env

    for path in _KEY_FILE_CANDIDATES:
        try:
            key = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if key:
            return key
    return ""


def auth_headers() -> dict:
    """给 requests / urllib 用的鉴权头；没配置密钥时返回空 dict（兼容老部署）。"""
    key = api_key()
    return {"X-API-Key": key} if key else {}
