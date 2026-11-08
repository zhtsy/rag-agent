# -*- coding: utf-8 -*-
"""retrieval_cache.py 的测试。

缓存的坑几乎都不在"能不能存"，而在三件事：
  1. key 归一化——多一个空格就当新问题，命中率归零；
  2. 配置漂移——改了门槛却吃旧缓存，等于拿旧结论骗自己；
  3. 开关失效——评测时忘关，分数看着变快其实变假。
这三条各有一个断言守着。
"""

import pytest

import retrieval_cache as rc


# ---------- key 归一化 ----------

def test_whitespace_and_case_do_not_change_key():
    """多敲空格、大小写不同，是同一个问题，不能算两次。"""
    a = rc.make_key("小智喜欢什么", "")
    b = rc.make_key("  小智喜欢什么  ", "")
    c = rc.make_key("小智喜  欢什么", "")   # 中间多空格
    assert a == b == c


def test_different_queries_have_different_keys():
    assert rc.make_key("小智喜欢什么", "") != rc.make_key("小智住在哪", "")


def test_original_query_participates_in_key():
    """改写句相同、原话不同，实体过滤的依据就不同，必须分开存。"""
    assert rc.make_key("他喜欢什么", "小智喜欢什么") != rc.make_key("他喜欢什么", "小智住在哪")


def test_config_change_invalidates_key(monkeypatch):
    """改了门槛/窗口，旧缓存就不可信了。key 里带配置签名，自动换。"""
    monkeypatch.setenv("RAG_MIN_RELEVANT", "0.15")
    k1 = rc.make_key("问题", "")
    monkeypatch.setenv("RAG_MIN_RELEVANT", "0.30")
    k2 = rc.make_key("问题", "")
    assert k1 != k2


# ---------- TTL ----------

def test_ttl_expiry(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(rc.time, "time", lambda: now[0])
    cache = rc.TTLCache(ttl=10)
    cache.put("k", "v")
    assert cache.get("k") == "v"

    now[0] += 9          # 还没到
    assert cache.get("k") == "v"
    now[0] += 2          # 过了
    assert cache.get("k") is None


def test_expired_entry_is_removed_not_just_hidden(monkeypatch):
    """过期要就地删，否则 size 只涨不跌，看监控会以为是命中率问题。"""
    now = [1000.0]
    monkeypatch.setattr(rc.time, "time", lambda: now[0])
    cache = rc.TTLCache(ttl=5, maxsize=10)
    cache.put("k", "v")
    now[0] += 6
    cache.get("k")
    assert cache.stats()["size"] == 0


# ---------- LRU ----------

def test_lru_evicts_oldest(monkeypatch):
    cache = rc.TTLCache(ttl=100, maxsize=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_get_refreshes_recency(monkeypatch):
    """读过的条目不先淘汰——否则热点和冷数据一样被挤掉。"""
    cache = rc.TTLCache(ttl=100, maxsize=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")           # a 变最近
    cache.put("c", 3)        # 该淘汰 b
    assert cache.get("a") == 1
    assert cache.get("b") is None


# ---------- 开关 ----------

def test_disabled_cache_never_hits(monkeypatch):
    """RAG_CACHE=0 时必须完全不读不写：评测就靠这个保证分数真实。"""
    monkeypatch.setenv("RAG_CACHE", "0")
    rc.reset()
    rc.store("q", "", "结果")
    assert rc.lookup("q", "") is None


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
def test_disabled_values(monkeypatch, value):
    monkeypatch.setenv("RAG_CACHE", value)
    assert rc.enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_enabled_values(monkeypatch, value):
    monkeypatch.setenv("RAG_CACHE", value)
    assert rc.enabled() is True


def test_enabled_store_then_hit(monkeypatch):
    monkeypatch.setenv("RAG_CACHE", "1")
    rc.reset()
    rc.store("小智喜欢什么", "", "【来源：test1.txt】\n喜欢打篮球")
    assert rc.lookup("  小智喜欢什么 ", "") == "【来源：test1.txt】\n喜欢打篮球"
    assert rc.get_cache().stats()["hits"] == 1


def test_empty_result_is_not_stored():
    """空串不入缓存：那是异常路径的残渣，存下来会把一次失败钉死 5 分钟。"""
    rc.reset()
    rc.store("q", "", "")
    assert rc.lookup("q", "") is None


# ---------- 统计 ----------

def test_stats_shape():
    s = rc.get_cache().stats()
    for key in ("size", "hits", "misses", "hit_rate", "ttl", "maxsize", "enabled"):
        assert key in s
