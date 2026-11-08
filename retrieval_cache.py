# -*- coding: utf-8 -*-
"""检索结果缓存（TTL + LRU，纯标准库）。

为什么缓存在检索层，而不是缓最终答案
------------------------------------
- 缓答案：首字延迟变成"整条答完"（流式没了），而且会把上一轮的语气/格式一起冻住。
- 缓检索：只跳过 graph / retrieve / rerank（实测 3~8s 里的 2.5~4s），
  LLM 仍然按**当前**上下文流式生成，答案还是这一轮的。

为什么跨会话复用不会串台
------------------------
检索只吃 query（实体过滤用的是用户本轮原话），与对话历史无关；
历史影响的是生成阶段，不在这一层。所以 key 只用 query。

评测为什么要关掉
----------------
缓存会掩盖检索耗时与波动，让分层归因失真。评测脚本默认 `RAG_CACHE=0`，
这是刻意的：**先有可信的分层评测，才允许加缓存**，顺序不能反。
"""

import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

DEFAULT_TTL = 300      # 秒。知识库更新后最多 5 分钟内自愈。
DEFAULT_MAXSIZE = 256  # 条。单条结果是几个文本块，256 条约几 MB。

_FALSY = {"0", "false", "no", "off", ""}


def enabled() -> bool:
    """RAG_CACHE=0/false/no/off 关闭；不设或其它值开启。"""
    return os.environ.get("RAG_CACHE", "1").strip().lower() not in _FALSY


def normalize(text: str) -> str:
    """只做与语义无关的归一：去掉全部空白、转小写。

    为什么是"去掉"而不是"压缩成一个空格"：中文里 `小智喜 欢什么` 和
    `小智喜欢什么` 是同一个问题，压缩只会保留那个空格、变成两个 key。
    英文多词问题去掉空格后依然彼此可区分，不会串。
    """
    return "".join((text or "").split()).lower()


def _env_signature() -> str:
    """影响检索结果的运行期配置。

    改了门槛/窗口却不失效缓存，会拿旧判定骗自己。把这两个值并进 key，
    配置一变自动换 key，不需要人工清缓存。
    """
    return "|".join(
        [
            os.environ.get("RAG_MIN_RELEVANT", ""),
            os.environ.get("RAG_RERANK_CANDIDATES", ""),
        ]
    )


def make_key(query: str, original_query: str) -> str:
    raw = "\x1f".join([normalize(query), normalize(original_query), _env_signature()])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TTLCache:
    """带 TTL 的 LRU。线程安全：检索工具跑在线程池里，会被并发调用。"""

    def __init__(self, ttl: int = DEFAULT_TTL, maxsize: int = DEFAULT_MAXSIZE) -> None:
        self.ttl = ttl
        self.maxsize = maxsize
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            ts, value = item
            if now - ts > self.ttl:
                # 过期就地删，避免 get 命中但返回旧值
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._data),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "ttl": self.ttl,
                "maxsize": self.maxsize,
                "enabled": enabled(),
            }


def _ttl_from_env() -> int:
    try:
        return max(0, int(os.environ.get("RAG_CACHE_TTL", DEFAULT_TTL)))
    except (TypeError, ValueError):
        return DEFAULT_TTL


_search_cache = TTLCache(ttl=_ttl_from_env())


def get_cache() -> TTLCache:
    return _search_cache


def lookup(query: str, original_query: str) -> Optional[str]:
    """命中返回缓存字符串，未命中/关闭返回 None。"""
    if not enabled():
        return None
    return _search_cache.get(make_key(query, original_query))


def store(query: str, original_query: str, result: str) -> None:
    if not enabled():
        return
    if not isinstance(result, str) or not result:
        return
    _search_cache.put(make_key(query, original_query), result)


def reset() -> None:
    """测试用：清空并重置计数。"""
    global _search_cache
    _search_cache = TTLCache(ttl=_ttl_from_env())
