# -*- coding: utf-8 -*-
"""索引目录解析的测试。

这一层的价值全在「配错时不出事」：切实验索引配错路径，最坏的结局不是报错，
而是静默连上一个空库——检索全空、全部拒答，看起来像模型坏了。
所以每个回落分支都要有用例钉住。
"""

import os

import pytest

from kb_paths import DEFAULT_DIRNAME, ENV_VAR, looks_like_index, resolve


def _make_index(path):
    """造一个"像索引"的目录：chroma 只要这个文件在，就说明写过数据。"""
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, "chroma.sqlite3"), "w").close()
    return path


def test_env_unset_uses_default(tmp_path):
    d, note = resolve(str(tmp_path), {})
    assert d == os.path.join(str(tmp_path), DEFAULT_DIRNAME)
    assert note == ""


def test_env_blank_treated_as_unset(tmp_path):
    d, note = resolve(str(tmp_path), {ENV_VAR: "   "})
    assert d == os.path.join(str(tmp_path), DEFAULT_DIRNAME)
    assert note == ""


def test_relative_env_is_joined_to_base(tmp_path):
    _make_index(tmp_path / "chroma_db_v3")
    d, note = resolve(str(tmp_path), {ENV_VAR: "chroma_db_v3"})
    assert d == os.path.join(str(tmp_path), "chroma_db_v3")
    assert note == ""


def test_absolute_env_used_as_is(tmp_path):
    target = _make_index(tmp_path / "elsewhere" / "idx")
    d, note = resolve(str(tmp_path / "base"), {ENV_VAR: str(target)})
    assert d == str(target)
    assert note == ""


def test_missing_dir_falls_back_and_explains(tmp_path):
    d, note = resolve(str(tmp_path), {ENV_VAR: "no_such_dir"})
    assert d == os.path.join(str(tmp_path), DEFAULT_DIRNAME)
    assert "不存在" in note


def test_empty_dir_falls_back(tmp_path):
    """目录在、但没有 chroma.sqlite3 ＝ 空库，必须按不可用处理。"""
    os.makedirs(tmp_path / "empty_idx")
    d, note = resolve(str(tmp_path), {ENV_VAR: "empty_idx"})
    assert d == os.path.join(str(tmp_path), DEFAULT_DIRNAME)
    assert "sqlite3" in note


def test_env_pointing_to_default_dir_is_silent(tmp_path):
    _make_index(tmp_path / DEFAULT_DIRNAME)
    d, note = resolve(str(tmp_path), {ENV_VAR: DEFAULT_DIRNAME})
    assert d == os.path.join(str(tmp_path), DEFAULT_DIRNAME)
    assert note == ""


@pytest.mark.parametrize("name", ["chroma_db", "chroma_db_v3"])
def test_project_indexes_are_recognised(name):
    """真实仓库里的两套索引都要被判为可用（路径不存在则跳过，CI 没有索引）。"""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), name)
    if not os.path.isdir(path):
        pytest.skip(f"{name} 不存在（CI 不建索引）")
    assert looks_like_index(path)
