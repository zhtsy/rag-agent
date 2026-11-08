# -*- coding: utf-8 -*-
"""evidence.py 的单元测试。

样本全部来自 `_search_knowledge_base_impl` / `search_knowledge_graph` 的真实输出格式，
不是凭空构造的 JSON——这一层的价值就在于"对得上真实字符串"，
用编造的样本测等于没测。

零第三方依赖（只用 pytest），可以进那个 0.2 秒的 CI。
"""

import pytest

from evidence import format_evidence, parse_retrieval


# ----------------------------------------------------------------------
# 1. 命中：向量 / 混合检索
# ----------------------------------------------------------------------
class TestEvidence:
    def test_single_block(self):
        """最基本的形态：一条证据、带来源和 rerank 分。"""
        raw = (
            "【来源：knowledge_base/人物志.md】【置信度：0.9312】\n"
            "小智住在南京，喜欢猫。"
        )
        p = parse_retrieval(raw)
        assert p["kind"] == "evidence"
        assert len(p["hits"]) == 1
        assert p["hits"][0]["source"] == "knowledge_base/人物志.md"
        assert p["hits"][0]["score"] == pytest.approx(0.9312)
        assert p["hits"][0]["content"] == "小智住在南京，喜欢猫。"
        assert p["hits"][0]["kind"] == "vector"

    def test_multiple_blocks_keep_order(self):
        """多块按原始顺序返回——顺序就是 rerank 的排序，打乱等于毁掉排序信息。"""
        raw = (
            "【来源：a.md】【置信度：0.95】\n第一块\n\n---\n\n"
            "【来源：b.md】【置信度：0.80】\n第二块\n\n---\n\n"
            "【来源：c.md】【置信度：0.61】\n第三块"
        )
        p = parse_retrieval(raw)
        assert p["kind"] == "evidence"
        assert [h["source"] for h in p["hits"]] == ["a.md", "b.md", "c.md"]
        assert [round(h["score"], 2) for h in p["hits"]] == [0.95, 0.80, 0.61]

    def test_graph_fact_block_has_no_score(self):
        """因果类问题会把图谱确定性事实插在最前，它没有 rerank 分。

        这条要钉住 score is None——如果给它补一个假的 1.0，
        下游算平均置信度时会被系统性抬高。
        """
        raw = (
            "【来源：知识图谱属性】【确定性事实】\n"
            "加里奥由禁魔石制成；雕塑家杜朗制造了加里奥\n\n---\n\n"
            "【来源：knowledge_base/加里奥.md】【置信度：0.4942】\n"
            "加里奥能吸收魔法。"
        )
        p = parse_retrieval(raw)
        assert p["kind"] == "evidence"
        assert p["hits"][0]["kind"] == "graph_fact"
        assert p["hits"][0]["score"] is None
        assert p["hits"][1]["score"] == pytest.approx(0.4942)

    def test_multiline_content_preserved(self):
        """块正文可能是多段（原文里就有换行），不能被截成第一行。"""
        raw = "【来源：a.md】【置信度：0.9】\n第一段。\n\n第二段。"
        p = parse_retrieval(raw)
        assert p["hits"][0]["content"] == "第一段。\n\n第二段。"

    def test_separator_inside_content_needs_blank_lines(self):
        """正文里出现单换行的 --- 不该被当成分隔符。

        分隔符是 `\\n\\n---\\n\\n`（空行包裹），markdown 分割线常是单换行，
        两者混淆会把一段正文劈成两半，凭空多出一条"证据"。
        """
        raw = "【来源：a.md】【置信度：0.9】\n上\n---\n下"
        p = parse_retrieval(raw)
        assert p["kind"] == "evidence"
        assert len(p["hits"]) == 1
        assert "上\n---\n下" in p["hits"][0]["content"]

    @pytest.mark.parametrize("score_text,expected", [
        ("1.0000", 1.0),
        ("0.0000", 0.0),
        ("0.4942", 0.4942),
    ])
    def test_score_boundaries(self, score_text, expected):
        raw = f"【来源：a.md】【置信度：{score_text}】\n内容"
        assert parse_retrieval(raw)["hits"][0]["score"] == pytest.approx(expected)


# ----------------------------------------------------------------------
# 2. 图谱路径：置信度在末尾，且没有来源标签
# ----------------------------------------------------------------------
class TestGraphAnswer:
    def test_tail_score_is_extracted(self):
        """图谱返回 `事实句【置信度：0.9123】`，置信度在尾部。

        这和向量的头部格式是两套写法，曾经差点把两者当同一种解析——
        结果 content 里残留了 `【置信度：...】` 字样，模型照着念了出来。
        """
        raw = "小智住在南京，喜欢猫。【置信度：0.9123】"
        p = parse_retrieval(raw)
        assert p["kind"] == "graph_answer"
        assert p["hits"][0]["score"] == pytest.approx(0.9123)
        assert p["hits"][0]["content"] == "小智住在南京，喜欢猫。"
        assert "【置信度" not in p["hits"][0]["content"]

    def test_graph_kind_is_distinguishable_from_vector(self):
        """两种 confidence 语义不同，kind 必须能区分。

        向量的 0.93 是 rerank 分，图谱的 0.93 是 Cypher 事实可信度。
        合并统计没有意义，所以这里要求 kind 不同。
        """
        vector = parse_retrieval("【来源：a.md】【置信度：0.93】\n内容")
        graph = parse_retrieval("内容【置信度：0.93】")
        assert vector["kind"] != graph["kind"]
        assert vector["hits"][0]["kind"] != graph["hits"][0]["kind"]


# ----------------------------------------------------------------------
# 3. 图谱命中：头部来源 + 中间事实 + 尾部置信度（三段式）
# ----------------------------------------------------------------------
class TestGraphEvidence:
    def test_real_graph_shape(self):
        """真实形态，取自线上 /search 的实测输出。

        这条是被真实数据打脸之后补的：最初的实现只认「事实句 + 尾部置信度」，
        把头部标签当成了正文的一部分，于是返回给调用方的 content 里
        赫然写着 `【图谱检索结果】【来源：test.txt】`——标签泄漏进了正文。

        教训：解析层不能凭对源码的印象写，必须拿真实输出校准。
        """
        raw = "【图谱检索结果】【来源：test.txt】\n小智住在南京。【置信度：0.9994】"
        p = parse_retrieval(raw)
        assert p["kind"] == "graph_evidence"
        h = p["hits"][0]
        assert h["source"] == "test.txt"
        assert h["score"] == pytest.approx(0.9994)
        assert h["content"] == "小智住在南京。"
        assert "【图谱检索结果】" not in h["content"]
        assert "【来源：" not in h["content"]

    def test_multiple_sources(self):
        """图谱命中可能横跨多个文档，来源是逗号分隔的一串。"""
        raw = "【图谱检索结果】【来源：a.txt, b.txt】\n事实。【置信度：0.8】"
        assert parse_retrieval(raw)["hits"][0]["source"] == "a.txt / b.txt"

    def test_no_confidence_when_scoring_failed(self):
        """reranker 打分失败时上游不补假分数，这一层也不许拿 0 或 1 兜底。

        伪造一个数值比留空更有害：留空调用方知道"没有分"，
        伪造则让下游把它当成真实分数参与统计。
        """
        raw = "【图谱检索结果】【来源：a.txt】\n事实句。"
        p = parse_retrieval(raw)
        assert p["kind"] == "graph_evidence"
        assert p["hits"][0]["score"] is None

    def test_multiline_facts_preserved(self):
        """图谱会把「结构化事实句」和「润色补充句」用换行拼起来，两行都要留。"""
        raw = "【图谱检索结果】【来源：a.txt】\n结构化事实句。\n润色补充句。【置信度：0.7】"
        assert parse_retrieval(raw)["hits"][0]["content"] == "结构化事实句。\n润色补充句。"

    def test_empty_source_placeholder(self):
        """一个来源都没记到时上游写的是空串，这里要给可读占位而不是空字符串。"""
        raw = "【图谱检索结果】【来源：】\n事实。【置信度：0.5】"
        assert parse_retrieval(raw)["hits"][0]["source"] == "本地知识图谱"

    def test_format_shows_source_but_not_label(self):
        raw = "【图谱检索结果】【来源：test.txt】\n小智住在南京。【置信度：0.9994】"
        out = format_evidence(parse_retrieval(raw))
        assert "来源：test.txt" in out
        assert "0.9994" in out
        assert "【图谱检索结果】" not in out


# ----------------------------------------------------------------------
# 4. 未命中（正常结果）vs 出错（故障）——必须严格分开
# ----------------------------------------------------------------------
class TestNoMatchVersusError:
    @pytest.mark.parametrize("raw", [
        "未在知识库中找到相关信息。",
        "图谱中未找到相关信息。",
        "  未在知识库中找到相关信息。\n",   # 前后空白要能容忍
    ])
    def test_no_match(self, raw):
        assert parse_retrieval(raw)["kind"] == "no_match"

    @pytest.mark.parametrize("raw", [
        "检索工具执行失败，错误原因: connection reset",
        "检索问题为空，请重新描述你的问题。",
        "图谱检索出错: Neo4j 连接失败",
    ])
    def test_error(self, raw):
        assert parse_retrieval(raw)["kind"] == "error"

    def test_graph_error_is_not_read_as_no_match(self):
        """回归：图谱连不上 ≠ 知识库没有。

        两者对用户的答复完全不同：前者该重试或告警，后者该如实说"没这条资料"。
        混为一谈会把一次数据库故障伪装成一个正常拒答，线上看不出来。
        """
        p = parse_retrieval("图谱检索出错: ServiceUnavailable")
        assert p["kind"] == "error"
        assert "ServiceUnavailable" in p["message"]

    def test_empty_is_error(self):
        assert parse_retrieval("")["kind"] == "error"
        assert parse_retrieval("   ")["kind"] == "error"
        assert parse_retrieval(None)["kind"] == "error"


# ----------------------------------------------------------------------
# 4. 兜底：认不出结构时不许假装成功
# ----------------------------------------------------------------------
class TestFallback:
    def test_unstructured_text_falls_back(self):
        """认不出的输出要原样交出并标注，不能返回空 hits 冒充"解析成功"。"""
        p = parse_retrieval("这是一段没有来源标签的说明文字")
        assert p["kind"] == "text"
        assert p["raw"].startswith("这是一段")
        assert p["hits"][0]["content"] == "这是一段没有来源标签的说明文字"

    def test_partial_parse_is_rejected(self):
        """一半块能解析、一半不能 → 整体退回兜底。

        半解析比不解析更危险：调用方会以为拿到的是全部证据，
        然后基于"少了一截的上下文"作答。
        """
        raw = "【来源：a.md】【置信度：0.9】\n内容\n\n---\n\n没有来源标签的第二块"
        p = parse_retrieval(raw)
        assert p["kind"] == "text"
        assert len(p["hits"]) == 1              # 兜底只给一条原文
        assert "没有来源标签的第二块" in p["hits"][0]["content"]

    def test_raw_always_preserved(self):
        """raw 永远不丢：解析错了还能拿到原文排查。"""
        raw = "【来源：a.md】【置信度：0.9】\n内容"
        assert parse_retrieval(raw)["raw"] == raw


# ----------------------------------------------------------------------
# 5. 渲染：给上层 agent 的指令必须相反
# ----------------------------------------------------------------------
class TestFormat:
    def test_no_match_forbids_retry(self):
        """没命中时要明确禁止"换个说法重试"。

        不加这句，上层 agent 会把一次"知识库没有"放大成五六次无效检索，
        既浪费时间，也会让用户觉得系统答不出来还在硬撑。
        """
        out = format_evidence(parse_retrieval("未在知识库中找到相关信息。"))
        assert "不是服务故障" in out
        assert "反复检索" in out

    def test_error_permits_retry(self):
        """出错时要明确允许重试——与 no_match 的指令相反。"""
        out = format_evidence(parse_retrieval("检索工具执行失败，错误原因: timeout"))
        assert "可以重试" in out
        assert "反复检索" not in out

    def test_evidence_shows_source_and_score(self):
        raw = "【来源：knowledge_base/人物志.md】【置信度：0.9312】\n小智住在南京。"
        out = format_evidence(parse_retrieval(raw), elapsed_ms=1234, cached=False)
        assert "knowledge_base/人物志.md" in out
        assert "0.9312" in out
        assert "小智住在南京" in out
        assert "1234 ms" in out
        assert "未命中缓存" in out

    def test_cached_flag_rendered(self):
        raw = "【来源：a.md】【置信度：0.9】\n内容"
        assert "命中缓存" in format_evidence(parse_retrieval(raw), cached=True)

    def test_graph_fact_labelled_as_certain(self):
        """确定性事实不能显示成"相关度 0.xx"——它没有分，标出来才对。"""
        raw = "【来源：知识图谱属性】【确定性事实】\n加里奥由禁魔石制成"
        out = format_evidence(parse_retrieval(raw))
        assert "确定性事实" in out
        assert "相关度" not in out
