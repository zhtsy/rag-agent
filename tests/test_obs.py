# -*- coding: utf-8 -*-
"""obs.py 的测试（请求级耗时埋点）。

这个模块本身不该出错：它跑在每个请求的路径上，一旦抛异常就会把正常问答带崩。
所以测试重点是"失败也不能影响主流程"。
"""

import json
import time

import pytest

import obs


def test_stage_accumulates_and_counts():
    """同名阶段重复进入要累加耗时、单独记次数（模型偶尔一轮调两次检索）。"""
    trace = obs.Trace("测试问题")
    with trace.stage("retrieve"):
        time.sleep(0.01)
    with trace.stage("retrieve"):
        time.sleep(0.01)

    assert trace.calls["retrieve"] == 2
    assert trace.stages["retrieve"] >= 15      # 两次 10ms，允许调度误差


def test_add_records_manual_timing():
    """手工埋点（没法用 with 包住的既有循环）也要能累加、能计次。"""
    trace = obs.Trace()
    trace.add("llm", 120.0)
    trace.add("llm", 30.0)
    assert trace.stages["llm"] == 150.0
    assert trace.calls["llm"] == 2


def test_stage_records_time_even_on_exception():
    """异常路径也要记耗时："失败得慢"往往比"成功得慢"更值得查。"""
    trace = obs.Trace()
    with pytest.raises(ValueError):
        with trace.stage("rerank"):
            raise ValueError("boom")
    assert "rerank" in trace.stages


def test_finish_emits_one_json_line(capsys):
    trace = obs.start("根据知识库，马斯克是谁")
    with trace.stage("retrieve"):
        pass
    trace.note(conf=None, status="out_of_scope", sources=0)
    record = obs.finish(trace)

    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1                     # 一行就是一个 JSON，采集侧才好解析
    parsed = json.loads(out[0])
    assert parsed["ev"] == "req"
    assert parsed["rid"] == trace.rid
    assert "total_ms" in parsed and "stages" in parsed
    assert parsed["status"] == "out_of_scope"
    assert record["rid"] == trace.rid


def test_finish_keeps_none_notes_out_of_the_record(capsys):
    """None 值的 note 不写进日志：JSON 里出现 "conf": null 会被误读成"算出来是空"。"""
    trace = obs.start("q")
    trace.note(conf=None, sources=2)
    record = obs.finish(trace)
    assert "conf" not in record
    assert record["sources"] == 2


def test_finish_none_is_noop():
    assert obs.finish(None) is None


def test_stage_outside_request_does_not_crash():
    """脚本 / 单测里没有 start() 也不能炸——空转即可。"""
    with obs.stage("retrieve"):
        pass
    assert obs.current() is not None


def test_snapshot_does_not_leak_question():
    """SSE 终帧会下发 snapshot，里面不能带问题原文（那是用户输入）。"""
    trace = obs.start("我的身份证号是多少")
    snap = trace.snapshot()
    assert set(snap) == {"rid", "total_ms", "stages"}
    assert "身份证" not in json.dumps(snap, ensure_ascii=False)


def test_percentile():
    values = [float(i) for i in range(1, 101)]      # 1..100
    assert obs.percentile(values, 50) == 50.5
    # 线性插值算出 95.05，round(95.05, 1) 因浮点表示落到 95.0——
    # 差 0.05 毫秒，不值得为它改实现（真要改就得引 decimal）。
    assert obs.percentile(values, 95) in (95.0, 95.1)
    assert obs.percentile([0.0, 10.0], 50) == 5.0
    assert obs.percentile([], 95) is None
    assert obs.percentile([7.0], 95) == 7.0
