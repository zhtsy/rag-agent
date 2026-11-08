# -*- coding: utf-8 -*-
"""obs.py —— 请求级可观测性：request_id + 分阶段耗时 + 一行 JSON 日志。

为什么要有它
────────────
"RAG 慢在哪"是这套系统被问得最多、也最难答的问题。原来只有首字延迟一行日志，
答不出"是路由慢、检索慢、rerank 慢，还是模型生成慢"，于是所有性能优化都靠猜
（以及一次次重启压测）。2026-11-04 排查 rerank 窗口时，就是因为没有分阶段打点，
才先误判成"第二跳被丢"，绕了一大圈。

设计取舍
────────
- **只用标准库**。不引入 OpenTelemetry：这套系统跑在 2 vCPU 的机器上，
  otel 的 batch exporter 本身就是几百 MB 的内存和一条常驻线程，收益不匹配。
  这里就一个 ContextVar + 一个 perf_counter，零依赖、零后台线程。
- **直接 print 到 stdout**，不走 logging。容器里 loguru 的配置随时会变，
  而 stdout 是采集侧唯一稳定的契约（docker logs / 云日志都能直接收）。
  一行一个 JSON，可以直接 grep、jq、喂给聚合脚本。
- **ContextVar 而不是全局变量**。并发下每个请求各有一份 Trace，
  不会互相覆盖；asyncio 任务创建时会自动复制上下文，所以跨 await 也有效。
- 没开 trace 时（`RAG_TRACE_LOG=0`）所有操作退化为空转，不产生任何日志噪声。

用法
────
    from obs import start, finish, stage
    trace = start(question)
    with stage("rerank"):
        ...
    trace.note(sources=3, conf=0.92)
    finish(trace)

日志样例（真正的单行，这里为可读性折行）：
    {"ev":"req","rid":"9f3c1a2b7d4e","total_ms":2861.4,"stages":{"route":712.0,
     "retrieve":486.3,"rerank":1035.8,"llm":612.1},"ttft_ms":2408.0,
     "sources":2,"conf":0.9966,"status":"answered"}
"""

import contextvars
import json
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# 开关：默认开。RAG_TRACE_LOG=0 可整体关掉（评测跑批时嫌吵就关）。
_ENABLED = os.environ.get("RAG_TRACE_LOG", "1").strip().lower() not in ("0", "false", "no", "off")
# 可选：同时落一份 JSONL 到文件，供 trace_stats.py 算 P50/P95。
_FILE = os.environ.get("RAG_TRACE_FILE", "").strip()

QUESTION_MAX = 40   # 日志里只留问题前缀：够定位，又不会把长文档灌进日志


class Trace:
    """一次请求的计时容器。同名阶段可重复进入，耗时**累加**、次数单独记。"""

    __slots__ = ("rid", "question", "t0", "stages", "calls", "notes")

    def __init__(self, question: str = "") -> None:
        self.rid = uuid.uuid4().hex[:12]
        self.question = (question or "").strip()
        self.t0 = time.perf_counter()
        self.stages: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}
        self.notes: Dict[str, Any] = {}

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)

    def add(self, name: str, ms: float) -> None:
        """手工记一段耗时（毫秒）。

        用来处理"没法用 with 包住"的既有代码——例如一个巨大的 async for 循环，
        为了计时去缩进整段代码不值得，取两个时间戳相减再调它就行。
        """
        self.stages[name] = round(self.stages.get(name, 0.0) + ms, 1)
        self.calls[name] = self.calls.get(name, 0) + 1

    @contextmanager
    def stage(self, name: str):
        """`with trace.stage("rerank"):` —— 记录一段耗时（毫秒）。

        用 finally 记时，即使抛出异常这段也会被记上（异常路径的耗时同样是信息，
        而且"失败得慢"往往比"成功得慢"更值得查）。
        """
        began = time.perf_counter()
        try:
            yield self
        finally:
            self.add(name, (time.perf_counter() - began) * 1000)

    def note(self, **kwargs: Any) -> None:
        """往这条 trace 上挂任意标量（sources 数、置信度、top1 分、候选块数…）。"""
        self.notes.update(kwargs)

    def snapshot(self) -> Dict[str, Any]:
        """给 SSE 终帧 / 评测报告用的结构（不含问题原文，避免回传用户输入）。"""
        return {
            "rid": self.rid,
            "total_ms": self.elapsed_ms(),
            "stages": dict(self.stages),
        }


# 没有活动 trace 时的兜底：让 stage() 永远能安全调用，且不产生任何日志
_DETACHED = Trace()

_current: contextvars.ContextVar = contextvars.ContextVar("rag_trace", default=None)


def start(question: str = "") -> Trace:
    """开始一次请求的计时，并绑定到当前上下文（同请求内的代码都能拿到）。"""
    trace = Trace(question)
    _current.set(trace)
    return trace


def current() -> Trace:
    """取当前请求的 Trace；不在请求内（脚本、单测）时返回一个不进日志的空壳。"""
    return _current.get() or _DETACHED


@contextmanager
def stage(name: str):
    """模块级快捷方式：`with obs.stage("retrieve"): ...`"""
    trace = _current.get()
    if trace is None:
        yield _DETACHED          # 没有请求上下文，纯粹空转
    else:
        with trace.stage(name):
            yield trace


def _emit(record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    print(line, flush=True)
    if _FILE:
        try:
            with open(_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass                  # 落盘失败不能影响主流程，日志而已


def finish(trace: Optional[Trace], **extra: Any) -> Optional[Dict[str, Any]]:
    """收尾：把这条 trace 打成一行 JSON。返回记录本身，方便调用方复用。

    extra 会覆盖 trace.notes 里的同名键（调用方拿到的信息更新）。
    """
    if trace is None or not _ENABLED:
        return None

    record: Dict[str, Any] = {
        "ev": "req",
        "rid": trace.rid,
        "q": trace.question[:QUESTION_MAX],
        "total_ms": trace.elapsed_ms(),
        "stages": dict(trace.stages),
    }
    if any(c > 1 for c in trace.calls.values()):
        # 只在"确实重复调用过"时才带上次数：模型偶尔会一轮调两次检索，
        # 不标出来会让人误以为单次检索变慢了。
        record["calls"] = dict(trace.calls)
    record.update({k: v for k, v in trace.notes.items() if v is not None})
    record.update({k: v for k, v in extra.items() if v is not None})
    _emit(record)
    return record


def percentile(values: List[float], p: float) -> Optional[float]:
    """线性插值分位数（p 取 0~100）。空列表返回 None。

    故意不引 numpy：这套系统只为了算几个 P50/P95 装个 100MB 的包不划算。
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    pos = (len(ordered) - 1) * (max(0.0, min(100.0, p)) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return round(ordered[lo] * (1 - frac) + ordered[hi] * frac, 1)
