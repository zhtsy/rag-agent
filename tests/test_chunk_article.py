# -*- coding: utf-8 -*-
"""法条类文档必须按「条」切块，不能按字数装箱。

事故：真实法律语料上「《专利法》里的『发明』是指什么」12 条定义式题只拿 0.333。
文件找对了，但定义所在的那一段没进 top-5 —— 定长 200 字装箱把第一条（立法目的）
和第二条（定义）塞进同一块，向量被前半段稀释。法条自带编号，本来有明确语义边界，
按字数切断属于把文档结构扔掉了。
"""
from ingest.chunker import _article_of, chunk_doc
from ingest.loaders import Doc

LAW = """# 中华人民共和国专利法

## 第一章 总则

第一条 为了保护专利权人的合法权益，鼓励发明创造，制定本法。

第二条 本法所称的发明创造是指发明、实用新型和外观设计。

发明，是指对产品、方法或者其改进所提出的新的技术方案。

实用新型，是指对产品的形状、构造或者其结合所提出的适于实用的新的技术方案。

第三条 国务院专利行政部门负责管理全国的专利工作。

第四条 申请专利的发明创造涉及国家安全或者重大利益需要保密的，按照国家有关规定办理。

第五条 对违反法律、社会公德或者妨害公共利益的发明创造，不授予专利权。
"""


def _chunks():
    return chunk_doc(Doc(LAW, {"source_file": "专利法.md"}), size=200, overlap=40)


def test_each_article_is_its_own_chunk():
    chunks = _chunks()
    arts = [c.metadata.get("article") for c in chunks if c.metadata.get("article")]
    # 一条一块，且不重复、不漏
    assert arts == ["第一条", "第二条", "第三条", "第四条", "第五条"]


def test_definition_stays_with_its_article():
    """定义段（「发明，是指…」）必须和它所属的「第二条」在同一块。"""
    second = [c for c in _chunks() if c.metadata.get("article") == "第二条"]
    assert len(second) == 1
    text = second[0].text
    assert "本法所称的发明创造" in text
    assert "发明，是指对产品、方法" in text


def test_no_leak_between_articles():
    """重叠不能让上一条的尾巴混进下一条。"""
    for c in _chunks():
        if c.metadata.get("article") == "第二条":
            assert "为了保护专利权人" not in c.text


def test_body_carries_law_and_article():
    """续块与首块都要能自证出处：法名 + 章 + 条。"""
    text = [c for c in _chunks() if c.metadata.get("article") == "第三条"][0].text
    assert "中华人民共和国专利法" in text
    assert "第一章" in text
    assert "第三条" in text


def test_long_article_splits_with_prefix_repeated():
    long_body = "".join(f"第{n}项事项应当依照本法关于登记与公告的规定办理，不得另行约定。"
                        for n in range(1, 9))
    doc = Doc("# 示例法\n\n## 第一章 总则\n\n" +
              "\n\n".join([f"第{i}条 这是用于占位的条文。" for i in range(1, 6)]) +
              f"\n\n第六条 {long_body}\n", {"source_file": "示例法.md"})
    chunks = chunk_doc(doc, size=200, overlap=40)
    sixth = [c for c in chunks if c.metadata.get("article") == "第六条"]
    assert len(sixth) >= 2
    for c in sixth:
        assert "示例法" in c.text and "第六条" in c.text


def test_non_article_doc_keeps_old_packing():
    """只有零星「第 N 条」的普通文档不能误判成法条，仍按预算装箱。"""
    doc = Doc("# 报销制度\n\n" + "\n\n".join(
        f"这是第{i}段说明，长度填充到二十个字左右以便装箱测试。" for i in range(1, 9)
    ), {"source_file": "policy.md"})
    chunks = chunk_doc(doc, size=200, overlap=40)
    assert not any(c.metadata.get("article") for c in chunks)


def test_article_detector():
    assert _article_of("第三十六条　发明专利的申请人请求实质审查") == "第三十六条"
    assert _article_of("第一百二十条之一 附加刑") == "第一百二十条之一"
    assert _article_of("第三条不是开头") == ""
    assert _article_of("") == ""
