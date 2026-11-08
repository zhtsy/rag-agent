# -*- coding: utf-8 -*-
"""eval_layers 的测试。

这里的每条用例都对应一次**真实排查**：分不清"检索没找到"和"找到了没写进去"，
就只能靠人逐条读答案猜。所以断言的重点不是覆盖率，而是**归因不许串味**：

  · 证据在、答案没写 → 必须记 generation_loss，不许记成 retrieval_miss
  · 证据不在、答案也没有 → 必须记 retrieval_miss
  · 拒答类不能因为 "没有" 不在证据里就被判成检索漏召回
"""

import eval_layers as el


# ------------------------------------------------------------------ 头部剥离
def test_header_is_not_evidence():
    """`【来源：…】【置信度：0.9966】` 是元信息，不是证据正文。

    真实坑：某次给数字类用例设 must_contain=["9966"]，结果因为置信度头部
    恰好写着 0.9966，被判定为"证据里出现过"——归因直接指向生成层，查错了方向。
    """
    ctx = ["【来源：./knowledge_base/test.txt】【置信度：0.9966】\n小智喜欢玩游戏。"]
    body = el.context_body(ctx)
    assert "小智" in body
    assert "0.9966" not in body
    assert "knowledge_base" not in body


def test_header_strip_tolerates_multi_bracket_and_missing_header():
    """图谱块是 `【来源：知识图谱属性】【确定性事实】\n正文`，两段头部连写。"""
    assert el.strip_header("【来源：知识图谱属性】【确定性事实】\n加里奥 由...制成 禁魔石") == (
        "加里奥 由...制成 禁魔石"
    )
    # 没有头部的裸文本不能被删掉第一行
    assert el.strip_header("只有一行正文") == "只有一行正文"


# ------------------------------------------------------------------ 四种归因
def test_generation_loss_when_evidence_present_but_answer_omits():
    """证据里有、答案没写 → 生成层的问题，不是检索的。"""
    case = {"category": "直接检索", "must_contain": ["橘猫", "代码"]}
    ctx = ["【来源：test5.txt】【置信度：0.99】\n我养了一只橘猫，叫代码。"]
    row = el.attribute(case, "他养了一只猫。", ctx)
    assert row["layer"] == "generation_loss"
    assert row["lost_in_generation"] == ["橘猫", "代码"]
    assert row["lost_in_retrieval"] == []
    # 召回是满的：证据里两个词都在
    assert row["recall"] == 1.0


def test_retrieval_miss_when_evidence_absent():
    """证据里压根没有 → 召回的问题。这是马斯克那类题的真实归因。"""
    case = {"category": "直接检索", "must_contain": ["马斯克"]}
    ctx = ["【来源：test2.txt】【置信度：0.0026】\n我是小王，喜欢古风。"]
    row = el.attribute(case, "知识库中没有马斯克的信息。", ctx)
    assert row["layer"] == "retrieval_miss"
    assert row["lost_in_retrieval"] == ["马斯克"]
    assert row["recall"] == 0.0


def test_mixed_when_both_layers_fail():
    """一个词没召回、另一个词召回了没写——两层的锅都要背。"""
    case = {"category": "多跳推理", "must_contain": ["禁魔石", "泰拉瑞亚"]}
    ctx = ["【来源：test1.txt】【置信度：0.9】\n加里奥由禁魔石制成。"]
    row = el.attribute(case, "加里奥能吸收魔法。", ctx)
    assert row["layer"] == "mixed"
    assert row["lost_in_retrieval"] == ["泰拉瑞亚"]
    assert row["lost_in_generation"] == ["禁魔石"]


def test_violation_wins_over_everything():
    """答了不该答的，即使要点齐全也不算"答对"。

    T028「最拿手的菜」就是这种：问到的那道菜答对了，但把相邻的菜一起报了出来。
    """
    case = {"category": "跨主题", "must_contain": ["开水白菜"], "must_not_contain": ["回锅肉"]}
    ctx = ["【来源：test8.txt】\n他最拿手的是开水白菜。"]
    row = el.attribute(case, "开水白菜，也会做回锅肉。", ctx)
    assert row["layer"] == "violation"
    assert row["violation_words"] == ["回锅肉"]


# ------------------------------------------------------------------ 两类特例
def test_abstain_case_is_not_judged_on_recall():
    """拒答类的 must_contain 是"未找到/无法"，本来就不该出现在证据里。

    如果照普通口径算，每条拒答用例都会被判成 retrieval_miss——
    归因表会被 4 条"假缺陷"污染，真缺陷反而被淹没。
    """
    case = {"category": "拒答", "must_contain": ["未找到", "无法"], "must_not_contain": ["123456"]}
    row = el.attribute(case, "无法从知识库中找到相关信息。", ["【来源：test3.txt】\n我是小赵。"])
    assert row["layer"] == "n/a"
    assert row["recall"] is None


def test_abstain_case_still_flags_leak():
    """但拒答类要是真把密码编出来了，必须记 violation，绝不能被 n/a 吞掉。"""
    case = {"category": "拒答", "must_contain": ["无法"], "must_not_contain": ["123456"]}
    row = el.attribute(case, "密码是 123456。", [])
    assert row["layer"] == "violation"
    assert row["violation_words"] == ["123456"]


def test_needs_inference_marked_separately():
    """"南京"→"江苏" 这种推理题，关键词按字面永远搜不到，不该算检索的锅。

    只有出题人知道某个词是不是必须推出来的，所以这个标记来自用例元数据，
    不许在代码里塞启发式（"没搜到就当成推理题"会掩盖真正的召回缺陷）。
    """
    case = {"category": "图谱-多跳", "must_contain": ["江苏"], "needs_inference": True}
    row = el.attribute(case, "资料只提到南京，无法确定省份。", ["【来源：graph】\n小陈住在南京。"])
    assert row["layer"] == "inference_miss"


def test_same_case_without_flag_is_retrieval_miss():
    case = {"category": "图谱-多跳", "must_contain": ["江苏"]}
    row = el.attribute(case, "资料只提到南京。", ["【来源：graph】\n小陈住在南京。"])
    assert row["layer"] == "retrieval_miss"


# ------------------------------------------------------------------ 召回率口径
def test_unsupported_hit_does_not_inflate_recall():
    """模型靠自身记忆答对了、证据里却没有——不能算检索的功劳。"""
    case = {"category": "直接检索", "must_contain": ["python"]}
    row = el.attribute(case, "他写 Python。", [])
    assert row["layer"] == "ok"
    assert row["recall"] == 0.0
    assert row["unsupported_hits"] == ["python"]


def test_recall_is_case_insensitive():
    """模型写 `Python`、用例写 `python`，历史上被判成漏答过（T001 掉到 0.50）。"""
    case = {"category": "直接检索", "must_contain": ["python"]}
    assert el.attribute(case, "他用 Python 写脚本。", ["Python 是他的主力语言。"])["recall"] == 1.0


def test_abstention_restating_the_entity_is_not_a_hit():
    """拒答会把问题里的实体原样复述一遍，那不是"答到了"。

    真实场景：问"马斯克是谁"，回答"知识库中没有马斯克的信息"。
    按字面 `"马斯克" in answer` 判定为命中 → 归因判 ok、召回率 1.0，
    一个漏召回的问题被伪装成满分。这条断言就是为了让它回不去。
    """
    case = {"category": "直接检索", "must_contain": ["马斯克"]}
    row = el.attribute(case, "知识库中没有马斯克的信息。", [])
    assert row["in_answer"] == []
    assert row["missing_answer"] == ["马斯克"]
    assert row["recall"] == 0.0
    assert row["layer"] == "retrieval_miss"


# ------------------------------------------------------------------ 汇总与渲染
def _rows():
    return [
        {"id": "A", "score": 1.0, "recall": 1.0, "layer": "ok",
         "missing_answer": [], "lost_in_retrieval": [], "lost_in_generation": []},
        {"id": "B", "score": 0.0, "recall": 0.0, "layer": "retrieval_miss",
         "missing_answer": ["禁魔石"], "lost_in_retrieval": ["禁魔石"], "lost_in_generation": []},
        {"id": "C", "score": 0.5, "recall": 1.0, "layer": "generation_loss",
         "missing_answer": ["橘猫"], "lost_in_retrieval": [], "lost_in_generation": ["橘猫"]},
        {"id": "D", "score": 0.0, "recall": None, "layer": "violation",
         "missing_answer": [], "lost_in_retrieval": [], "lost_in_generation": []},
    ]


def test_summarize_splits_lost_score_by_layer():
    s = el.summarize(_rows())
    assert s["total"] == 4
    assert s["avg_score"] == 0.375
    assert s["lost_total"] == 2.5          # B(1.0) + C(0.5) + D(1.0)
    assert s["lost_retrieval"] == 1.0      # B
    assert s["lost_generation"] == 0.5     # C
    assert s["lost_violation"] == 1.0      # D
    # 三项之和必须等于总分损失，否则说明有分数的丢失没被归因到任何一层
    assert (
        s["lost_retrieval"] + s["lost_generation"] + s["lost_violation"] + s["lost_inference"]
        == s["lost_total"]
    )
    # 召回率只统计有 recall 的行（拒答类不计入）
    assert s["recall_n"] == 3
    assert s["retrieval_recall"] == 0.6667


def test_summarize_ignores_error_rows():
    rows = _rows() + [{"id": "E", "score": 0.0, "layer": "error", "recall": None}]
    s = el.summarize(rows)
    assert s["total"] == 5 and s["success"] == 4 and s["failed"] == 1
    # 失败的行不参与均分：否则一次网络抖动就把整体分数拉低
    assert s["avg_score"] == 0.375


def test_render_markdown_is_a_table():
    s = el.summarize(_rows())
    md = el.render_markdown(_rows(), s)
    assert "| 检索召回率 |" in md
    assert "`retrieval_miss`" in md
    # 缺陷明细里必须能看见"证据里也没有"这几个词，它决定了排查方向
    assert "证据里也没有" in md
    assert "禁魔石" in md


def test_every_layer_has_a_label():
    """新增标签忘了写人话，报告里就会出现生僻英文——渲染前就能发现。"""
    for name in el.DEFECT_LAYERS + ("ok", "n/a", "error"):
        assert name in el.LAYERS
