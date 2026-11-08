# -*- coding: utf-8 -*-
"""tests/test_breed_hierarchy.py —— 品种层级的三条不变量

守的不是"映射表写全了"（那改不了），而是**调用方会踩的坑**：
  1. 类别必须是合法类名，不能补出 `橘猫 -> 猫科动物` 这种没人查的边。
  2. 品种不能等于类别（自环），否则变长路径会绕死。
  3. 只补两端都在图里的边——这条最重要，它保证脚本不会凭空造节点。
"""

import pytest

from breed_hierarchy import (
    BREED_TO_CATEGORY,
    CATEGORY_NAMES,
    Edge,
    category_of,
    plan_edges,
    summary,
)


class TestInvariants:
    def test_every_category_is_a_known_name(self):
        for breed, cat in BREED_TO_CATEGORY.items():
            assert cat in CATEGORY_NAMES, f"{breed} 的类别 {cat} 不在 CATEGORY_NAMES 里"

    def test_自映射条目允许存在但绝不产生边(self):
        # 「仓鼠→仓鼠」是合法的：它自己就是类别。写进表里是为了表达"这是顶层"，
        # 但它必须被 plan_edges 的自环过滤掉，不能真的补一条 仓鼠→仓鼠。
        self_mapped = [b for b, c in BREED_TO_CATEGORY.items() if b == c]
        assert self_mapped, "至少要有一个自映射样本，否则这条测试是空转"
        for breed in self_mapped:
            edges, _ = plan_edges([breed])
            assert all(not e.is_self_ref for e in edges), f"{breed} 补出了自环"

    def test_no_key_has_whitespace(self):
        for breed in BREED_TO_CATEGORY:
            assert breed == breed.strip(), f"{breed!r} 带空白，会让查表漏命中"

    def test_summary_covers_all_entries(self):
        assert sum(n for _, n in summary()) == len(BREED_TO_CATEGORY)


class TestCategoryOf:
    @pytest.mark.parametrize(
        "breed,expect",
        [
            ("橘猫", "猫"),
            ("猫", None),          # "猫"本身没写进映射表，它是类别不是品种
            ("金毛", "狗"),
            ("哈士奇", "狗"),
            ("仓鼠", "仓鼠"),
        ],
    )
    def test_known(self, breed, expect):
        assert category_of(breed) == expect

    def test_whitespace_and_case_insensitive(self):
        assert category_of("  橘猫 ") == "猫"
        assert category_of("橘猫　") == "猫"   # 全角空格

    def test_unknown_returns_none_not_guess(self):
        # 史莱姆是 Terraria 里的怪物，不该被硬塞进任何类别
        assert category_of("史莱姆") is None
        assert category_of("") is None
        assert category_of(None) is None


class TestPlanEdges:
    def test_类别节点缺失时仍出边但记入missing(self):
        # 图里有 橘猫/猫/金毛，没有 狗 → 两条边都要出，`狗` 由 missing 报告给调用方去建。
        # 这是刻意设计：边和节点是两件事，缺节点不该静默丢边。
        edges, missing = plan_edges(["橘猫", "猫", "金毛"])
        got = {(e.breed, e.category) for e in edges}
        assert got == {("橘猫", "猫"), ("金毛", "狗")}
        assert missing == ["狗"]

    def test_品种节点不在图里就完全不出边(self):
        # 只有类别节点、没有品种节点时，不该凭空补边
        edges, missing = plan_edges(["狗", "猫"])
        assert edges == [] and missing == []

    def test_no_self_loop_for_a_breed_that_is_a_category(self):
        # 拿铁的品种就是"猫"；即便有人把"猫"当品种传进来，也不能补自环
        edges, _ = plan_edges(["猫", "猫"])
        assert all(not e.is_self_ref for e in edges)

    def test_empty_graph_plans_nothing(self):
        edges, missing = plan_edges([])
        assert edges == [] and missing == []

    def test_unknown_nodes_are_ignored(self):
        edges, _ = plan_edges(["史莱姆", "敌怪", "蓝史莱姆"])
        assert edges == []

    def test_display_name_keeps_original_spelling(self):
        edges, _ = plan_edges([" 橘猫 "])
        assert edges and edges[0].breed == "橘猫"


class TestEdge:
    def test_self_ref_flag(self):
        assert Edge("猫", "猫").is_self_ref is True
        assert Edge("橘猫", "猫").is_self_ref is False

    def test_str_is_readable(self):
        assert "橘猫" in str(Edge("橘猫", "猫"))
