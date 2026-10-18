# test_eval_system.py
"""
eval_system.py 的单元测试。
运行方式：pytest test_eval_system.py -v
"""
import json
import os
import pytest
from unittest.mock import patch, MagicMock

# 导入被测函数
from eval_system import evaluate


@pytest.fixture
def sample_test_cases():
    """提供一份标准测试用例数据"""
    return [
        {
            "id": "T001",
            "question": "小智喜欢什么？",
            "expected_answer": "小智喜欢Python",
            "category": "直接检索",
            "must_contain": ["Python", "游戏"],
            "must_not_contain": ["小赵"],
        },
        {
            "id": "T002",
            "question": "小智的银行卡密码是多少？",
            "expected_answer": "无法从知识库中找到",
            "category": "拒答",
            "must_contain": ["未找到", "无法"],
            "must_not_contain": ["123456"],
        },
    ]


@pytest.fixture
def fake_answer():
    """模拟一个 AgentAnswer 对象"""
    mock = MagicMock()
    mock.answer = "小智喜欢Python和游戏"
    mock.sources = ["./knowledge_base/test.txt"]
    mock.confidence = 0.95
    return mock


class TestEvaluatePipeline:
    """测试 evaluate() 的核心逻辑"""

    @patch("eval_system.time.sleep")  # 跳过 3 秒延迟
    @patch("eval_system.run_conversation")
    def test_evaluate_normal_case(self, mock_run, mock_sleep, tmp_path, sample_test_cases, fake_answer):
        """正常路径：命中关键词 + 无禁止词 → 得高分"""
        mock_run.return_value = fake_answer

        # 准备测试用例文件
        (tmp_path / "test_cases1.json").write_text(
            json.dumps(sample_test_cases[:1], ensure_ascii=False),
            encoding="utf-8",
        )

        # 切换工作目录（evaluate 用的是相对路径）
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            evaluate()
        finally:
            os.chdir(old_cwd)

        # 验证报告
        report = json.loads((tmp_path / "eval_report0.json").read_text(encoding="utf-8"))
        assert report["total"] == 1
        assert report["avg_score"] == 1.0  # Python 和 游戏 都命中，无禁止词
        assert report["details"][0]["hit_rate"] == 1.0
        assert report["details"][0]["violation_rate"] == 0.0

    @patch("eval_system.time.sleep")
    @patch("eval_system.run_conversation")
    def test_evaluate_keyword_partial_hit(self, mock_run, mock_sleep, tmp_path, fake_answer):
        """部分命中：must_contain 有两项，只命中一项 → 得 0.5"""
        mock_run.return_value = fake_answer  # 答案里只有 Python，没有"游戏"

        test_cases = [
            {
                "id": "T001",
                "question": "小智喜欢什么？",
                "expected_answer": "小智喜欢Python",
                "category": "直接检索",
                "must_contain": ["Python", "唱歌"],  # 唱歌不在答案里
                "must_not_contain": [],
            }
        ]
        (tmp_path / "test_cases1.json").write_text(
            json.dumps(test_cases, ensure_ascii=False),
            encoding="utf-8",
        )

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            evaluate()
        finally:
            os.chdir(old_cwd)

        report = json.loads((tmp_path / "eval_report0.json").read_text(encoding="utf-8"))
        assert report["avg_score"] == 0.5

    @patch("eval_system.time.sleep")
    @patch("eval_system.run_conversation")
    def test_evaluate_violation_reduces_score(self, mock_run, mock_sleep, tmp_path):
        """禁止词命中 → 得分被惩罚"""
        mock = MagicMock()
        mock.answer = "小智喜欢Python，小赵喜欢Java"  # 包含"小赵"，不包含"小王"
        mock.sources = ["test.txt"]
        mock.confidence = 0.9
        mock_run.return_value = mock

        test_cases = [
            {
                "id": "T001",
                "question": "小智喜欢什么？",
                "expected_answer": "小智喜欢Python",
                "category": "直接检索",
                "must_contain": ["Python"],
                "must_not_contain": ["小赵", "小王"],  # ← 只命中"小赵"，违规率 1/2
            }
        ]
        (tmp_path / "test_cases1.json").write_text(
            json.dumps(test_cases, ensure_ascii=False),
            encoding="utf-8",
        )

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            evaluate()
        finally:
            os.chdir(old_cwd)

        report = json.loads((tmp_path / "eval_report0.json").read_text(encoding="utf-8"))
        # 得分 = 1.0（关键词全中） * (1 - 0.5) = 0.5
        assert report["avg_score"] == 0.5

    @patch("eval_system.time.sleep")
    @patch("eval_system.run_conversation")
    def test_evaluate_refusal_case(self, mock_run, mock_sleep, tmp_path):
        """拒答类：命中任意一个关键词就算通过"""
        mock = MagicMock()
        mock.answer = "未找到相关信息"
        mock.sources = []
        mock.confidence = 0.0
        mock_run.return_value = mock

        test_cases = [
            {
                "id": "T002",
                "question": "小智的银行卡密码是多少？",
                "expected_answer": "无法从知识库中找到",
                "category": "拒答",
                "must_contain": ["未找到", "无法", "没有"],  # 命中"未找到"就算过
                "must_not_contain": ["123456"],
            }
        ]
        (tmp_path / "test_cases1.json").write_text(
            json.dumps(test_cases, ensure_ascii=False),
            encoding="utf-8",
        )

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            evaluate()
        finally:
            os.chdir(old_cwd)

        report = json.loads((tmp_path / "eval_report0.json").read_text(encoding="utf-8"))
        assert report["avg_score"] == 1.0

    @patch("eval_system.time.sleep")
    @patch("eval_system.run_conversation")
    def test_evaluate_string_result(self, mock_run, mock_sleep, tmp_path):
        """Agent 返回字符串（解析失败）时，不崩溃"""
        mock_run.return_value = "抱歉，我无法回答"  # 不是 AgentAnswer 对象

        test_cases = [
            {
                "id": "T003",
                "question": "某个问题",
                "expected_answer": "某个答案",
                "category": "直接检索",
                "must_contain": ["某个关键词"],
                "must_not_contain": [],
            }
        ]
        (tmp_path / "test_cases1.json").write_text(
            json.dumps(test_cases, ensure_ascii=False),
            encoding="utf-8",
        )

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            evaluate()  # 只要不抛异常就算通过
        finally:
            os.chdir(old_cwd)

        assert (tmp_path / "eval_report0.json").exists()