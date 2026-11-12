# -*- coding: utf-8 -*-
"""索引的「嵌入模型签名」：记录这个库是用哪个模型建的，加载时对一下。

为什么需要它
────────────
向量库只有"维度对得上"这一道隐式约束，**维度相同但模型不同**是完全合法的：
nomic-embed-text 和 bge-m3 都是 768 维，混用不会报错，只会让检索悄悄变差。
2026-11-12 实测过一次：法律语料用 nomic-embed-text（英文模型）建的库，
查「《专利法》里的『现有技术』是指什么」时，正确块的余弦 0.6921，
**低于**无关的第六十八条 0.7221 —— 排序是反的，检索必然落空，
而整条链路一声不响，只在评测里表现为"定义式 0.25"。

这类故障的特征是「不报错、只变差」，靠人盯着评测分数才发现，代价太高。
所以建库时把签名写进索引目录，加载时比对，不匹配就明确告警。

签名取 `provider:model`，例如 `ollama:bge-m3`。故意不含分数/维度/时间戳：
它要回答的问题只有一个 ——「查询和文档是不是同一个模型产生的」。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional

# 与 embeddings_shared 的默认值保持一致（那里 import langchain，CI 装不起，
# 所以这里不能直接引用；一致性由 tests/test_index_meta.py 钉住）。
DEFAULT_LOCAL_MODEL = "BAAI/bge-base-zh-v1.5"
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"

SIGNATURE_FILE = "_embed.json"

_dotenv_loaded = False


def _load_dotenv_once() -> None:
    """和 embeddings_shared 一样兜底读一次 .env。

    命令行脚本不一定经过 settings.py，而签名必须反映**实际生效**的配置，
    否则会拿默认值去比对，误报一堆不存在的冲突。
    """
    global _dotenv_loaded
    if _dotenv_loaded or os.getenv("RAG_EMBED_PROVIDER"):
        _dotenv_loaded = True
        return
    _dotenv_loaded = True
    env_file = Path(__file__).resolve().with_name(".env")
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        pass


def current_signature(env: Optional[Mapping[str, str]] = None) -> str:
    """当前配置会产生的签名。env 参数只给测试注入用。"""
    if env is None:
        _load_dotenv_once()
        env = os.environ
    provider = (env.get("RAG_EMBED_PROVIDER") or "bge").strip().lower()
    default = DEFAULT_OLLAMA_MODEL if provider == "ollama" else DEFAULT_LOCAL_MODEL
    model = (env.get("RAG_EMBED_MODEL") or default).strip()
    return f"{provider}:{model}"


def read_signature(index_dir: str) -> Optional[str]:
    """索引里记录的签名；没有（老库）返回 None —— 那是"未知"，不是"不匹配"。"""
    path = os.path.join(index_dir or "", SIGNATURE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    sig = data.get("signature") if isinstance(data, dict) else None
    return sig.strip() if isinstance(sig, str) and sig.strip() else None


def write_signature(index_dir: str, signature: Optional[str] = None) -> str:
    """建库收尾时写入。写失败不该让建库整体失败（只是少了一道保险）。"""
    sig = signature or current_signature()
    path = os.path.join(index_dir, SIGNATURE_FILE)
    try:
        os.makedirs(index_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"signature": sig}, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    return sig


def check_signature(index_dir: str, current: Optional[str] = None) -> Dict[str, str]:
    """比对索引签名与当前配置。

    返回 {status, stored, current, note}，status ∈ ok / mismatch / unknown：
    * unknown —— 老库没有签名文件。不能当不匹配处理，否则一上线满屏告警。
    * mismatch —— 必须告警：查询和文档不在同一向量空间，检索结果不可信。
    """
    stored = read_signature(index_dir)
    cur = current or current_signature()
    if not stored:
        return {"status": "unknown", "stored": "", "current": cur,
                "note": "索引没有嵌入签名（老库），无法校验"}
    if stored != cur:
        return {"status": "mismatch", "stored": stored, "current": cur,
                "note": f"索引用 {stored} 建，当前是 {cur} —— 检索结果不可信"}
    return {"status": "ok", "stored": stored, "current": cur, "note": ""}
