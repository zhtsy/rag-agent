"""自包含问句判据：只有它成立时检索才跳过模型改写。

这个判据存在的唯一理由是**确定性**——改写每次措辞不同会让召回块换一批，
而 seed 只锁得住生成、锁不住检索。所以判据不能太宽（把追问也判成自包含，
代词就再也消解不了），也不能太窄（该跳的没跳，抖动照旧）。
"""

import rag_agent as ra


def test_self_contained_long_question():
    assert ra._query_is_self_contained("《耕地占用税法》所附平均税额表里，上海每平方米的平均税额是多少？")
    assert ra._query_is_self_contained("《商标法》里的「集体商标」是指什么？")


def test_pronoun_question_is_not_self_contained():
    # 代词问句必须交给改写，否则「它」永远匹配不到上一轮的实体
    assert not ra._query_is_self_contained("它的品种是什么？")
    assert not ra._query_is_self_contained("这个商标的注册条件有哪些？")


def test_elided_followup_is_not_self_contained():
    assert not ra._query_is_self_contained("那专利呢，也是这么规定的吗？")
    assert not ra._query_is_self_contained("还有别的注意事项吗？")


def test_too_short_is_not_self_contained():
    assert not ra._query_is_self_contained("税率多少")
    assert not ra._query_is_self_contained("")


def test_rewrite_mode_default_is_auto(monkeypatch):
    monkeypatch.delenv("RAG_QUERY_REWRITE", raising=False)
    assert ra._rewrite_mode() == "auto"
    monkeypatch.setenv("RAG_QUERY_REWRITE", "LLM")
    assert ra._rewrite_mode() == "llm"
