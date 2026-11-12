# -*- coding: utf-8 -*-
"""索引嵌入签名：库是用哪个模型建的，必须能验。

事故：用 nomic-embed-text（英文模型）建中文法律库，查「现有技术」时
正确块余弦 0.6921、无关的第六十八条 0.7221 —— 排序反了，检索必然落空。
但整条链路不报任何错：两个模型都是 768 维，Chroma 只校验维度。
"""
import json
import os

import pytest

import index_meta as im


def test_signature_shape():
    assert im.current_signature({}) == f"bge:{im.DEFAULT_LOCAL_MODEL}"
    assert im.current_signature({"RAG_EMBED_PROVIDER": "ollama"}) == \
        f"ollama:{im.DEFAULT_OLLAMA_MODEL}"
    assert im.current_signature({"RAG_EMBED_PROVIDER": "OLLAMA ",
                                 "RAG_EMBED_MODEL": "bge-m3"}) == "ollama:bge-m3"


def test_write_read_roundtrip(tmp_path):
    d = str(tmp_path / "idx")
    assert im.read_signature(d) is None          # 目录都不存在也不炸
    assert im.write_signature(d, "ollama:bge-m3") == "ollama:bge-m3"
    assert im.read_signature(d) == "ollama:bge-m3"
    with open(os.path.join(d, im.SIGNATURE_FILE), encoding="utf-8") as f:
        assert json.load(f)["signature"] == "ollama:bge-m3"


def test_check_status_three_way(tmp_path):
    ok_dir = str(tmp_path / "a")
    im.write_signature(ok_dir, "ollama:bge-m3")
    assert im.check_signature(ok_dir, current="ollama:bge-m3")["status"] == "ok"

    bad = im.check_signature(ok_dir, current="ollama:nomic-embed-text")
    assert bad["status"] == "mismatch"
    assert "nomic" in bad["note"] and "bge-m3" in bad["note"]

    # 老库没有签名文件：是「未知」，不能当不匹配（否则一上线满屏告警）
    assert im.check_signature(str(tmp_path / "nope"), current="x:y")["status"] == "unknown"


def test_broken_file_is_unknown_not_crash(tmp_path):
    d = str(tmp_path / "b")
    os.makedirs(d)
    with open(os.path.join(d, im.SIGNATURE_FILE), "w", encoding="utf-8") as f:
        f.write("{不是 json")
    assert im.read_signature(d) is None
    assert im.check_signature(d, current="a:b")["status"] == "unknown"


def test_defaults_match_embeddings_shared():
    """两处默认值必须是同一个常量，否则签名会把「一致」误报成「不一致」。"""
    es = pytest.importorskip("embeddings_shared")
    assert im.DEFAULT_LOCAL_MODEL == es.DEFAULT_MODEL
    assert im.DEFAULT_OLLAMA_MODEL == es.DEFAULT_OLLAMA_MODEL
