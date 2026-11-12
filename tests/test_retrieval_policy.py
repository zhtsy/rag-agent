# -*- coding: utf-8 -*-
"""检索选块策略的回归测试。

这个文件的目的不是"覆盖率高"，而是**把三次真实事故钉住**：
任何一个后来者把窗口调小、把加成写进置信度、或者改回"关键词硬占名额"，
这里必须立刻变红。

所以每个用例的文档字符串写的都是"哪次事故、当时的数字是多少"。
"""

import retrieval_policy as rp

# ========== 复现 T019 现场的真实数据 ==========
#
# 「加里奥为什么能吸收魔法」共 12 个候选。RRF 把它们排好后，含「禁魔石」的 3 块
# 落在下标 6/7/8；全量 rerank 后最好的一块（0.4942）整体排第 5。
# 当时窗口设成 6，下标 6 根本没被打分，于是 top-5 里只有那个 0.285 的低分块。
CONTENTS_12 = [
    "（加里奥的背景）加里奥是一名德玛西亚的禁卫军，他的职责是守护城门。",      # 0
    "（魔法奥能）魔法奥能是一种源自符文之地的能量形式，可以被特定存在吸收。",  # 1
    "（加里奥的雕像）加里奥最初是一尊石像，在危难中被唤醒。",                  # 2
    "（禁卫军职责）禁卫军负责巡逻与警戒，防止外敌入侵。",                      # 3
    "（德玛西亚）德玛西亚是一个崇尚禁魔的城邦，禁止使用魔法。",                # 4
    "（魔法奥能的性质）奥能会寻找能够承载它的容器，并非所有物质都能承载。",    # 5
    "（雕塑家杜朗）雕塑家杜朗用禁魔石雕刻了加里奥，禁魔石能吸收魔法。",        # 6 ← 关键
    "（禁魔石）禁魔石是一种能吸收并压制魔法的稀有矿石。",                      # 7
    "（杜朗的工艺）杜朗以禁魔石为材，使雕像天然具备抗魔特性。",                # 8
    "（唤醒仪式）据说加里奥在魔法侵袭中被唤醒，从此成为守护者。",              # 9
    "（符文之地）符文之地是魔法高度发达的大陆。",                              # 10
    "（其他）德玛西亚的军队以纪律严明著称。",                                  # 11
]

# 全量 12 个候选的 rerank 分（按上面顺序）。数值取自当时的实测结果：
# 前 4 名高于 0.5，第 5 名就是禁魔石那块 0.4942，再往下是 0.285 那块低分块。
SCORES_12 = [
    0.86, 0.71, 0.62, 0.53, 0.31, 0.285,
    0.4942, 0.44, 0.41, 0.33, 0.20, 0.18,
]

KEYWORDS_T019 = ["加里奥", "吸收", "魔法"]


def _pipeline(contents, scores, keywords, max_candidates, **kw):
    """模拟真实链路：先按窗口截断候选，只给窗口内的块打分，再选块。

    这正是 `search_knowledge_base` 的顺序 —— 截断在 rerank **之前**。
    """
    keep = rp.candidate_slice(len(contents), max_candidates)
    kept_contents = [contents[i] for i in keep]
    kept_scores = [scores[i] for i in keep]
    return rp.select(kept_contents, kept_scores, keywords, **kw), kept_contents


# ========== 事故 1：rerank 窗口砍到 6 ==========

def test_default_window_keeps_forbidden_stone():
    """默认窗口必须让「禁魔石」进 top-5。

    把 `DEFAULT_RERANK_CANDIDATES` 从 10 改回 6，这条立刻红 —— 这就是
    "哪个测试会挡住你"的答案。
    """
    sel, kept = _pipeline(
        CONTENTS_12, SCORES_12, KEYWORDS_T019, rp.DEFAULT_RERANK_CANDIDATES
    )
    picked = [kept[i] for i in sel.indices]
    assert len(picked) == 5
    assert any("禁魔石" in t for t in picked), "top-5 里必须有一块讲禁魔石"


def test_window_6_loses_forbidden_stone():
    """反证：窗口 6 时禁魔石根本不在候选里（负向对照，防止有人"优化"回去）。

    这条故意断言"缺陷存在"，用来记录当时的行为。如果哪天有人把截断改成
    "先打分后截断"，它会变红，那时应该删掉它而不是改数据。
    """
    sel, kept = _pipeline(CONTENTS_12, SCORES_12, KEYWORDS_T019, 6)
    picked = [kept[i] for i in sel.indices]
    assert not any("禁魔石" in t for t in picked)
    assert max(sel.raw_scores) == 0.86


def test_default_window_is_at_least_ten():
    """再钉一道常量：窗口只能更大，不能更小。"""
    assert rp.DEFAULT_RERANK_CANDIDATES >= 10


# ========== 事故 2：关键词硬占名额，高分被低分挤掉 ==========

def test_no_low_score_squeezes_out_high():
    """取前 5 必须纯按分数。

    旧写法是「前 4 个从关键词命中队列取，第 5 个从其他队列补」，
    于是第 5 高分（0.4942）被砍，换进 0.285 那个低分块。
    这里断言：选中的原始分就是全部候选里最高的 5 个，一个都不能被替换。
    """
    sel, kept = _pipeline(CONTENTS_12, SCORES_12, KEYWORDS_T019, 10)
    expected = sorted(SCORES_12[:10], reverse=True)[:5]
    assert sorted(sel.raw_scores, reverse=True) == expected
    assert 0.285 not in sel.raw_scores


def test_keyword_boost_only_breaks_near_ties():
    """加成只能做倾向，不能把明显低分的块顶上来。"""
    # 0.60 命中关键词（→0.65）、0.64 不命中：加成后翻盘，允许
    sel = rp.select(["这块没提那个词", "这块提了禁魔石"], [0.64, 0.60], ["禁魔石"])
    assert sel.indices == [1, 0]
    # 0.90 vs 0.60：加成后 0.65 仍然低于 0.90，顺序不变
    sel = rp.select(["这块没提那个词", "这块提了禁魔石"], [0.90, 0.60], ["禁魔石"])
    assert sel.indices == [0, 1]


# ========== 事故 3：加成后的分被当成置信度 ==========

def test_boost_never_contaminates_raw_score():
    """0.9966 加成后是 1.0466，但 raw 必须还是 0.9966。

    不然写进【置信度】的就是 1.0466，超过 1 会被下游 `<= 1.0` 过滤整条丢掉，
    满分题于是显示成 0.5（"无度量"）—— 实测「小智喜欢什么」就这样掉了。
    """
    sel = rp.select(["小智喜欢听古典音乐"], [0.9966], ["小智", "喜欢"])
    idx, sort_score, raw = sel.items[0]
    assert raw == 0.9966
    assert round(sort_score, 4) == 1.0466
    assert sel.raw_scores == [0.9966]


def test_selected_raw_is_never_above_one():
    """顺带保证：不管加成多少，raw 都不会越过 1.0（下游过滤的前提）。"""
    sel = rp.select(["x"], [1.0], ["x"])
    assert max(sel.raw_scores) <= 1.0


# ========== 相关性门槛 ==========

def test_out_of_scope_returns_nothing():
    """「马斯克是谁」实测 5 块分数 0.0026/0.0019/0.0019/0.0015/0.0010。

    一块都不能返回 —— 否则前端来源栏会挂上 test2/test3，用户以为那些文件写了答案。
    """
    sel = rp.select(
        ["a", "b", "c", "d", "e"],
        [0.0026, 0.0019, 0.0019, 0.0015, 0.0010],
        ["马斯克"],
    )
    assert sel.relevant is False
    assert sel.items == []
    assert sel.best_raw == 0.0026


def test_normal_question_passes_threshold():
    """正常问题最低实测 0.6078，必须过。"""
    sel = rp.select(["a", "b"], [0.6078, 0.31], ["小张", "小李"])
    assert sel.relevant is True
    assert len(sel.items) == 2


def test_threshold_boundary_is_inclusive():
    sel = rp.select(["a"], [rp.DEFAULT_MIN_RELEVANT], [])
    assert sel.relevant is True


# ========== 健壮性 ==========

def test_scalar_score_supported():
    """单候选时 compute_score 返回 float，必须能吃下（曾导致 zip 抛错）。"""
    sel = rp.select(["只有一块"], 0.87, ["只有"])
    assert sel.raw_scores == [0.87]
    assert sel.relevant is True


def test_length_mismatch_raises():
    """宁可显式报错，也不要静默按最长的那个 zip（会错位配对）。"""
    try:
        rp.select(["a", "b"], [0.5])
    except ValueError:
        return
    raise AssertionError("长度不一致必须抛 ValueError")


def test_empty_candidates():
    sel = rp.select([], [], [])
    assert sel.relevant is False and sel.items == [] and sel.best_raw == 0.0


def test_tie_order_is_stable():
    """同分按下标升序，保证两次运行结果一致（评测要可复现）。"""
    sel = rp.select(["a", "b", "c"], [0.5, 0.5, 0.5], [])
    assert sel.indices == [0, 1, 2]


def test_candidate_slice():
    assert list(rp.candidate_slice(12, 10)) == list(range(10))
    assert list(rp.candidate_slice(5, 10)) == list(range(5))
    # 0/负数按"不截断"处理 —— 这是**有意**和旧行为不同的一处：
    # 旧代码 `docs[:0]` 会把候选清空，于是不管问什么都说"未在知识库中找到"，
    # 配错一个环境变量就等于把检索关了。静默降级到"不截断"比整站哑掉强。
    assert list(rp.candidate_slice(5, 0)) == list(range(5))
    assert list(rp.candidate_slice(5, -3)) == list(range(5))


def test_extract_keywords_drops_stopwords_and_short_tokens():
    """停用词和单字词不能进加成词表。

    这里显式传 tokenizer：在线链路传的是 jieba，测试不该依赖它
    （`requirements-dev.txt` 里只有 pytest，CI 装 jieba 是浪费）。
    """
    fake_jieba = lambda t: ["根据", "知识库", "小陈", "养了", "什么", "宠物", "的"]
    kw = rp.extract_keywords("根据知识库，小陈养了什么宠物", fake_jieba)
    assert "小陈" in kw and "宠物" in kw
    assert "知识库" not in kw and "什么" not in kw and "的" not in kw


def test_extract_keywords_without_jieba_is_safe():
    """没装 jieba 时的兜底：不崩、不返回空，只是词边界粗。

    明确记录它的局限：中文按标点切，切不出"小陈"这种词。所以它只用于
    "环境里没有 jieba"的场景，代价是加成可能整体失效（不影响检索正确性，
    只影响打平局时的倾向）。
    """
    kw = rp.extract_keywords("根据知识库，小陈养了什么宠物", rp.fallback_tokenize)
    assert kw, "兜底 tokenizer 至少要给出非空词表"
    assert all(len(w) >= 2 for w in kw)
    assert "知识库" not in kw


def test_keyword_hit_count():
    assert rp.keyword_hit_count(["含小陈", "无关"], [0, 1], ["小陈"]) == 1


# ========== 窗口必须随索引规模放大 ==========
#
# 事故：语料从 145 块扩到 3196 块后，「苏州瑞运物流有限公司的对接人是谁」的
# 20 个候选里正确那块排第 12 位，被窗口 10 截在门外 → top1 只有 0.1096
# → 低于 0.15 门槛 → 判「检索落空」→ 拒答。检索其实找到了，是窗口太小。
def test_window_keeps_production_scale_unchanged():
    """生产规模（145 块）必须与原常量逐字一致，行为不能变。"""
    assert rp.candidate_window(0) == rp.DEFAULT_RERANK_CANDIDATES
    assert rp.candidate_window(145) == rp.DEFAULT_RERANK_CANDIDATES
    assert rp.candidate_window(300) == rp.DEFAULT_RERANK_CANDIDATES


def test_window_grows_with_index():
    assert rp.candidate_window(301) > rp.DEFAULT_RERANK_CANDIDATES
    assert rp.candidate_window(1000) >= 16
    assert rp.candidate_window(3196) >= 20


def test_window_never_exceeds_sane_bound():
    for n in (0, 145, 300, 1000, 3196):
        assert 1 <= rp.candidate_window(n) <= 24


# ========== 环境变量只能调大窗口，不能压过规模下限 ==========
#
# 事故：.env 里写死 RAG_RERANK_CANDIDATES=10，而代码是 "env or candidate_window(...)"，
# 环境变量永远为真 → 自适应窗口一次都没生效。7,000 块语料上日志仍是
# "rerank 候选截断: 20 → 10"，正确块落在第 11–20 名照样被截掉。
def test_env_override_cannot_shrink_below_scale_floor():
    assert rp.rerank_window(7082, "10") == 24      # 环境变量写 10 也压不下去
    assert rp.rerank_window(7082, 10) == 24


def test_env_override_can_enlarge():
    assert rp.rerank_window(7082, "32") == 32
    assert rp.rerank_window(145, "30") == 30


def test_env_unset_or_junk_falls_back_to_floor():
    for bad in (None, "", "  ", "abc", "0", -3):
        assert rp.rerank_window(7082, bad) == 24


def test_production_scale_unchanged_under_override():
    # 生产 145 块 + .env 的 10：和旧行为逐字一致
    assert rp.rerank_window(145, "10") == 10
