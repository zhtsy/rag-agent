# -*- coding: utf-8 -*-
"""
main1.py —— FastAPI 后端（SSE 流式 + 数据飞轮反馈采集）

关键点：
1. StreamingResponse + text/event-stream
2. headers 必须带 X-Accel-Buffering: no / Cache-Control: no-cache（否则反代会攒包）
3. 每次 yield 之后 await asyncio.sleep(0.01)，主动让出事件循环，逼 uvicorn 立刻下发
4. 【数据飞轮】/feedback 接口把点赞/踩 + 问题 + 回答 + ★检索上下文 落盘为 feedback_log.jsonl
5. 【证据链】/chat/stream 的收尾帧额外下发 contexts（原始文档块），前端随 👎 一起回传，
   离线归因脚本靠它做"事实比对"，而不是靠猜。

关于 /feedback 的线程安全：
    uvicorn 是多 worker/多协程的，append 写文件如果并发打开同一文件仍可能交错。
    这里用 threading.Lock 串行化写入 + 以 "a" 模式追加，保证"一行就是一条合法 JSON"。
    文件 IO 很轻（几 KB/次），但为了不阻塞事件循环，放进 asyncio.to_thread 里跑。
"""

import os
import json
import time
import uuid
import asyncio
import secrets
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from rag_agent import stream_conversation, aget_session_meta, search_knowledge_base
from api_auth import api_key as _configured_api_key
from answer_guard import completeness_supplement
import evidence as _evidence   # 检索字符串 → 结构化证据；/search 与 MCP 共用，零依赖可单测
import obs as _obs   # 请求级 request_id + 分阶段耗时，收尾打一行 JSON（见 obs.py）
import retrieval_cache as _retrieval_cache   # 检索结果缓存；/health 暴露它的统计

# 注意：import rag_agent 会加载重排模型，终端里看到 Loading weights 是正常的


# =====================================================================
# 0. 常量 / 路径
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent          # 项目根目录（不管从哪个 cwd 启动都稳）
FEEDBACK_LOG_PATH = BASE_DIR / "feedback_log.jsonl"  # 飞轮原始数据（JSONL，追加写）

# 单条反馈的字符上限，防止有人粘贴一整本书把日志撑爆
MAX_QUESTION_CHARS = 4000
MAX_ANSWER_CHARS = 20000
MAX_COMMENT_CHARS = 2000
# 上下文的体积上限：块数 + 单块字符 + 总字符，三重限制，防止日志文件爆炸
MAX_CONTEXTS = 12
MAX_CONTEXT_CHARS = 3000
MAX_CONTEXTS_TOTAL_CHARS = 24000

# 写文件的全局锁：同一个进程内串行化 append，避免多协程同时写导致半行 JSON
_FEEDBACK_WRITE_LOCK = threading.Lock()

# ---------------------------------------------------------------- 并发闸门
# 为什么需要：PERF.md 实测（2 vCPU 突发性能型）饱和点在**并发 4**，再往上
# 吞吐不再增长、只是排队变长——而 rerank 是唯一随并发恶化的一段（929ms → 1848ms），
# 因为它是纯 CPU、和 LLM 的 I/O 等待抢不到同一份资源。
# 没有上限时，8 个并发请求会让每一个都变慢，第 9 个依然全速挤进来，
# 最后所有人一起等，P95 一路涨到不可用。
#
# 策略：**快速失败而不是无限排队**。超过上限就短暂等一会，等不到就回 429，
# 让调用方 5 秒后重试——这比让用户干等 40 秒拿到答案更有用。
# 上限默认 4 = 实测饱和点；机器换成非突发型可以调大，改 RAG_MAX_CONCURRENT。
MAX_CONCURRENT_STREAMS = max(1, int(os.environ.get("RAG_MAX_CONCURRENT", "4")))
# 排队等待预算（秒）。超过就 429：等 8 秒还没轮到，说明已经过载。
MAX_QUEUE_WAIT_S = max(0.0, float(os.environ.get("RAG_QUEUE_WAIT", "8")))

# 信号量延迟到首次使用才创建：asyncio 原语要绑定到**正在运行的事件循环**，
# 在模块导入期创建（那时还没有 loop）在部分 Python 版本上会绑定错 loop。
_STREAM_SLOTS: Optional[asyncio.Semaphore] = None
_INFLIGHT = 0           # 仅供观测：当前正在流式输出的请求数

# ---- /search 自己的闸门（与流式名额分开）----
# 为什么不复用流式名额：/search 是纯 CPU（rerank），和流式请求抢的是同两个核。
# 借流式的名额会让批量的检索任务占满通道，把正在等答案的用户挤成 429。
# 默认 2 = 这台机器（2 vCPU）的物理上限：再多也只是互相拖慢，
# 而吞吐不会增加——和 RAG_MAX_CONCURRENT 取 4 是同一个饱和点推导。
MAX_SEARCH_CONCURRENT = max(1, int(os.environ.get("RAG_MAX_SEARCH_CONCURRENT", "2")))
_SEARCH_SLOTS: Optional[asyncio.Semaphore] = None


def _search_slots() -> asyncio.Semaphore:
    global _SEARCH_SLOTS
    if _SEARCH_SLOTS is None:
        _SEARCH_SLOTS = asyncio.Semaphore(MAX_SEARCH_CONCURRENT)
    return _SEARCH_SLOTS


def _slots() -> asyncio.Semaphore:
    global _STREAM_SLOTS
    if _STREAM_SLOTS is None:
        _STREAM_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)
    return _STREAM_SLOTS


def _release_stream_slot() -> None:
    """归还一个并发名额。必须是幂等的——生成器可能在多路径上关闭。"""
    global _INFLIGHT
    _INFLIGHT = max(0, _INFLIGHT - 1)
    sem = _STREAM_SLOTS
    if sem is not None:
        sem.release()


app = FastAPI(
    title="本地 RAG Agent API",
    description="基于 LangGraph + Qwen2.5 的本地知识库问答系统（真流式）",
    version="2.3.0",
)

# 允许跨域：Streamlit 走的是服务端 requests，本来不需要；
# 但如果你以后想用纯前端页面 / 浏览器直接调 /feedback，就不会被 CORS 挡住。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 内网自用；对外暴露时请改成具体域名
    allow_credentials=False,   # allow_origins=["*"] 时必须为 False
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# 1. 请求/响应模型
# =====================================================================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"


class SearchRequest(BaseModel):
    """纯检索请求体。

    query 必须自包含：检索层不做指代消解（那是对话链路的职责），
    所以"他住在哪"在这里查不到东西，调用方要先把"他"换成实体名。
    """
    query: str
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="最多返回几条证据。只能往下截——上游选块数量由检索策略决定，"
                    "要 20 条也只给得到实际选出来的那几条。",
    )


class FeedbackRequest(BaseModel):
    """前端点👍/👎时提交的反馈体。

    必填 4 个字段（题目要求）；后面几个是可选的"增强字段"，
    前端传了就一起落盘，不传完全不影响——这样接口对老前端是向后兼容的。
    """
    session_id: Optional[str] = "default_session"
    question: str = Field(..., description="用户当时提的问题（原样，不要清洗）")
    answer: str = Field(..., description="系统给出的完整回答")
    feedback: Literal["good", "bad"] = Field(..., description="只有 good / bad 两种")

    # ↓ 可选增强字段（便于离线归因时看到上下文）
    sources: Optional[List[str]] = Field(default=None, description="本轮命中的知识来源")
    contexts: Optional[List[str]] = Field(
        default=None,
        description="★ 本轮检索到的原始文档块（前端从 SSE 收尾帧里拿到后原样回传，归因脚本靠它做事实比对）",
    )
    confidence: Optional[float] = Field(default=None, description="后端给出的置信度")
    comment: Optional[str] = Field(default=None, description="用户补充文字说明（可以没有）")
    message_id: Optional[str] = Field(default=None, description="前端消息唯一 ID，用于去重")


# =====================================================================
# 2. 基础探活接口
# =====================================================================
@app.get("/")
def root():
    return {"message": "RAG Agent API is running", "status": "healthy"}


@app.get("/health")
def health():
    # 带上并发水位：光 status=ok 看不出"已经排满了"，
    # 而排满时接口依然是 200（还在服务），监控会以为一切正常。
    #
    # 也带上缓存统计。理由不是"顺便看看"，而是让**评测能自证**：
    # 跑分时如果 cache.enabled=true 且 hits>0，那份耗时就不代表冷路径，
    # 分数也只能整体信、不能拿去和"无缓存基线"比。
    return {
        "status": "ok",
        "inflight": _INFLIGHT,
        "limit": MAX_CONCURRENT_STREAMS,
        "cache": _retrieval_cache.get_cache().stats(),
    }


def _sse(payload: dict) -> str:
    """统一的 SSE 帧格式：data: {json}\n\n（必须两个换行，否则浏览器不会 flush）"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# =====================================================================
# 2.5 访问鉴权（只保护 /chat/stream）
# =====================================================================
def _presented_api_key(request: Request) -> str:
    """从请求头取密钥，两种写法都认：X-API-Key 优先，其次 Authorization: Bearer。"""
    value = (request.headers.get("x-api-key") or "").strip()
    if value:
        return value
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def require_api_key(request: Request) -> None:
    """`/chat/stream` 的鉴权依赖：密钥不对直接 401，**在加载任何模型/检索之前**短路。

    为什么 fail-closed（密钥没配置就 503，而不是"没配就放行"）：
    8000 端口是挂在公网上的，放行等于"配错一次 = 永久无鉴权"，而且没人会察觉。
    宁可明确报错、让人立刻发现配置问题。

    key 每次请求都重新读文件（见 api_auth），所以轮换密钥不需要重启服务。
    """
    expected = _configured_api_key()
    if not expected:
        logger.error("🔐 后端未配置访问密钥（.rag_api_key 为空或不可读），已拒绝 /chat/stream")
        raise HTTPException(status_code=503, detail="服务端未配置访问密钥")

    got = _presented_api_key(request)
    if not got or not secrets.compare_digest(got, expected):
        client = request.client.host if request.client else "unknown"
        logger.warning(f"🚫 拒绝未授权请求 from {client}（密钥缺失或错误）")
        raise HTTPException(
            status_code=401,
            detail="无效或缺失的访问令牌（请在请求头带上 X-API-Key）",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# =====================================================================
# 3. 流式对话接口（完全保持原样，未做任何改动）
# =====================================================================
@app.post("/chat/stream", dependencies=[Depends(require_api_key)])
async def chat_stream_endpoint(request: ChatRequest):
    """
    流式接口（SSE）。

    鉴权：
      必须带 `X-API-Key: <密钥>`（或 `Authorization: Bearer <密钥>`）。
      密钥来源见 api_auth.py（优先环境变量 RAG_API_KEY，否则 .rag_api_key 文件）。
      校验发生在进入本函数之前：401 = 密钥缺失/错误，503 = 服务端没配密钥。

    帧协议：
      data: {"text": "..."}                        ← 正文增量
      data: {"text": "（补充：…）"}                 ← 完整性兜底追加（可能没有）
      data: {"sources": [...], "confidence": 0.9,  ← 收尾元信息
             "status": "answered", "contexts": [...]}
      data: [DONE]                                 ← 结束标记
      data: {"error": "..."}                       ← 异常

    confidence 语义（唯一口径，见 rag_agent._compute_confidence）：
      · null       本轮是拒答（知识库没有就明说），"置信度"这个量不适用
      · 0.0~1.0    本轮检索最相关那块证据的 rerank 分（越高越贴题）
      · 0.5        拿不到 rerank 分（无度量），不要当成"中等把握"来用
    拒答为什么是 null 而不是 0：0 会被读成"答砸了"，但正确拒答并没有答错。
    想知道"拒答得对不对"，看 status：
      · answered      有证据 + 正常作答
      · out_of_scope  检索没命中 + 拒答 → 知识库本来就没有，正常
      · missed        检索命中 + 拒答   → 漏答，要排查
      · ungrounded    检索没命中 + 硬答 → 无依据作答，幻觉风险
    confidence 不是"答对的概率"，别拿它做业务判断；质量监控请用 /feedback 的 bad 率。

    并发保护：
      同时在跑的流式请求上限 `RAG_MAX_CONCURRENT`（默认 4，= 实测饱和点）。
      排满后最多等 `RAG_QUEUE_WAIT` 秒（默认 8），等不到直接回 **429**
      （带 `Retry-After: 5`），而不是让请求无限排队把 P95 拖到不可用。
    """
    session_id = request.session_id or "default_session"
    logger.info(f"📥 [stream] 收到请求(session={session_id[:8]}): {request.message}")

    # ★ 请求级计时起点。在端点里 start()（而不是在生成器里），
    #   是因为 ContextVar 一旦在这条请求的上下文里设好，后面 create_task /
    #   线程池里的工具调用都自动继承——不用到处传参。
    trace = _obs.start(request.message)

    # ---- 并发闸门：先抢名额，抢不到就快速失败 ----
    # 放在 StreamingResponse 之前：一旦开始吐 header 就没法再改成 429 了。
    global _INFLIGHT
    wait_began = time.perf_counter()
    try:
        await asyncio.wait_for(_slots().acquire(), timeout=MAX_QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        waited_ms = (time.perf_counter() - wait_began) * 1000
        trace.add("queue", waited_ms)
        _obs.finish(trace, error="overloaded", inflight=_INFLIGHT)
        logger.warning(
            f"🚦 并发已满（{_INFLIGHT}/{MAX_CONCURRENT_STREAMS}），"
            f"排队 {waited_ms:.0f}ms 仍未轮到，回 429"
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": "服务繁忙，请稍后重试",
                "inflight": _INFLIGHT,
                "limit": MAX_CONCURRENT_STREAMS,
                "retry_after_s": 5,
            },
            headers={"Retry-After": "5"},
        )
    _INFLIGHT += 1
    queue_ms = (time.perf_counter() - wait_began) * 1000
    if queue_ms >= 50:
        # 只有真排过队才记：否则每次请求都多一个 0.3ms 的噪声阶段
        trace.add("queue", queue_ms)
        logger.info(f"⏳ 排队等待 {queue_ms:.0f}ms 才拿到并发名额")

    async def event_generator():
        # 先把响应头顶出去，避免浏览器/requests 等 header 时"卡住不动"
        yield ": connected\n\n"
        await asyncio.sleep(0.01)

        pieces: List[str] = []

        try:
            async for chunk in stream_conversation(request.message, session_id):
                if not chunk:
                    continue
                pieces.append(chunk)
                yield _sse({"text": chunk})
                # 关键：主动让出事件循环，逼 Uvicorn 把这一帧立刻 write 出去
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info("🔌 客户端已断开，流式响应终止")
            raise
        except Exception as e:
            logger.error(f"❌ 流式输出报错: {e}")
            logger.error(traceback.format_exc())
            _obs.finish(trace, error=type(e).__name__)
            yield _sse({"error": str(e)})
            await asyncio.sleep(0.01)
            return

        # ---- 收尾元信息（失败不影响正文）----
        meta = None
        try:
            meta = await aget_session_meta(session_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"⚠️ 读取收尾元信息失败: {e}")

        # ---- 完整性兜底：模型偶发丢掉原文限定成分（答"布丁"丢"仓鼠"）----
        # 不改写模型已吐出的正文，只在末尾补一句"来自检索原文"的说明。
        # 规则是确定性的（见 answer_guard.py），判错最多多一句原文事实，不会编造。
        try:
            extra = completeness_supplement("".join(pieces), (meta or {}).get("contexts") or [])
            if extra:
                logger.info(f"🧩 完整性兜底追加：{extra}")
                yield _sse({"text": extra})
                await asyncio.sleep(0.01)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"⚠️ 完整性兜底失败（忽略）: {e}")

        try:
            if meta:
                yield _sse({
                    "sources": meta.get("sources", []),
                    # 可能是 None：本轮拒答时"置信度"不适用（见 rag_agent._compute_confidence）。
                    # 不要用 0.0 兜底——0 会被读成"系统答得很差"，而拒答本身可能是完全正确的。
                    "confidence": meta.get("confidence"),
                    # 本轮属于哪种情况，前端据此决定显示"置信度 x%"还是"知识库无相关内容"
                    "status": meta.get("status", "answered"),
                    # ★ 新增：把检索到的原始文档块一起下发。
                    #   前端会把它挂在这条消息上，用户点 👎 时原样回传，归因脚本才有"证据"可比。
                    "contexts": meta.get("contexts", []),
                    # ★ 分阶段耗时（毫秒）：前端可以展示"检索 0.5s / 生成 0.6s"，
                    #   离线评测也能据此算 P95——不用再单独维护一套埋点。
                    "timings": trace.snapshot(),
                })
                await asyncio.sleep(0.01)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"⚠️ 下发收尾元信息失败: {e}")

        # 一行 JSON 汇总本轮：{rid, total_ms, stages{decide/retrieve/rerank/llm}, ttft_ms, ...}
        # 打在这里而不是在 finally 里，是为了让 total_ms 覆盖到最后一次 yield。
        _obs.finish(trace)

        yield "data: [DONE]\n\n"
        await asyncio.sleep(0.01)

    async def guarded_stream():
        """包一层只为**归还并发名额**。

        不把 try/finally 塞进 event_generator 里，是因为它内部有多处 `return`
        （异常、客户端断开），逐条路径去补释放一定会漏。包在最外层，
        正常结束、抛异常、被取消、被 GC 关闭——四种关闭方式都会走到 finally。
        """
        try:
            async for piece in event_generator():
                yield piece
        finally:
            _release_stream_slot()

    return StreamingResponse(
        guarded_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# =====================================================================
# 3.5 纯检索接口 —— 只要证据，不要生成
# =====================================================================
# 为什么这件事值得一个独立端点，而不是让调用方去消费 /chat/stream：
#
#   1. **拿得到证据本身**。走过 /chat/stream 拿到的是"模型读完证据后的复述"，
#      而评测归因、MCP 工具、外部 agent 要的都是原始证据。从答案反推证据是不可能的。
#   2. **省一次 LLM 调用**。检索 1~4s，生成再叠 2~8s。批处理场景下这个差价是决定性的。
#   3. **不占流式名额**。它有自己的闸门（见 MAX_SEARCH_CONCURRENT），批量的检索
#      不会把正在等答案的用户挤成 429。
#
# 返回 kind 的四种取值（语义见 evidence.py 文件头）：
#   evidence / graph_answer / no_match / error
# 注意：**no_match 返回 200 而不是 404**。"知识库里没有这条"是正常的业务结果，
# 不是"资源不存在"。用 404 会诱导调用方去重试一个本来就该拒答的问题。
@app.post("/search", dependencies=[Depends(require_api_key)])
async def search_endpoint(request: SearchRequest):
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")
    if len(query) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400, detail=f"query 过长（上限 {MAX_QUESTION_CHARS} 字符）"
        )

    began = time.perf_counter()
    try:
        await asyncio.wait_for(_search_slots().acquire(), timeout=MAX_QUEUE_WAIT_S)
    except asyncio.TimeoutError:
        logger.warning(f"🚦 [search] 检索通道已满，回 429")
        raise HTTPException(
            status_code=429,
            detail="检索繁忙，请稍后重试",
            headers={"Retry-After": "5"},
        )

    try:
        # 先探一次缓存，只为在响应里标出 cached。
        # 真正的读取仍是 search_knowledge_base 内部做的——这里不做拦截，
        # 是为了保住那层原有的 obs 埋点（cache=hit/miss）。
        cached = _retrieval_cache.lookup(query, query) is not None

        # 检索是同步的、且 rerank 吃满 CPU，必须挪到线程池，
        # 否则会把整个事件循环卡住——流式请求也会跟着停。
        text = await asyncio.to_thread(
            search_knowledge_base.invoke,
            {"query": query, "original_query": query},
        )
    finally:
        _search_slots().release()

    elapsed_ms = (time.perf_counter() - began) * 1000
    payload = _evidence.parse_retrieval(text)
    payload["hits"] = payload["hits"][: request.top_k]
    payload["query"] = query
    payload["cached"] = cached
    payload["elapsed_ms"] = round(elapsed_ms, 1)

    # raw 一并返回，不省这点带宽：
    # 解析逻辑一旦和上游拼装格式脱节（kind=text 就是信号），
    # 只有原文能定位问题。丢了 raw 就只能靠猜。
    logger.info(
        f"🔎 [search] {query!r} → {payload['kind']}，{len(payload['hits'])} 条，"
        f"{elapsed_ms:.0f}ms，cached={cached}"
    )
    return payload


# =====================================================================
# 4. 【新增】反馈接收接口 —— 数据飞轮的入口
# =====================================================================
def _append_feedback_line(record: dict) -> None:
    """把一条反馈以 JSONL 形式追加到文件末尾（同步函数，会在工作线程里被调用）。

    为什么用 "a" 追加模式：
        O_APPEND 语义下，单次 write 的偏移由内核保证在文件尾，
        再加上 _FEEDBACK_WRITE_LOCK，基本杜绝"两条 JSON 粘在同一行"。
    为什么 flush + fsync：
        飞轮数据丢了就白瞎了，落盘确认一下；量级只有每天几百条，开销可忽略。
    """
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)   # ensure_ascii=False：中文原样存，方便人肉看

    with _FEEDBACK_WRITE_LOCK:
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())   # 极端重视数据安全时可留；嫌慢可删掉这 2 行
            except OSError:
                pass


def _sanitize_contexts(contexts: Optional[List[str]]) -> List[str]:
    """上下文瘦身：限块数、限单块长度、限总长度。

    为什么要限：contexts 是原文，动辄几千字。不设上限的话，
    几百条反馈就能把 feedback_log.jsonl 撑到几百 MB，后面分析脚本读一次要几十秒。
    """
    if not contexts:
        return []

    out: List[str] = []
    used = 0
    for raw in contexts:
        if not isinstance(raw, str):
            raw = str(raw)
        text = raw.strip()
        if not text:
            continue
        text = text[:MAX_CONTEXT_CHARS]
        if used + len(text) > MAX_CONTEXTS_TOTAL_CHARS:
            break
        out.append(text)
        used += len(text)
        if len(out) >= MAX_CONTEXTS:
            break
    return out


def _read_feedback_stats() -> dict:
    """统计日志文件里的 good / bad 条数（坏行自动跳过），顺带统计"带证据"的覆盖率。"""
    if not FEEDBACK_LOG_PATH.exists():
        return {"total": 0, "good": 0, "bad": 0, "bad_with_context": 0}

    total = good = bad = bad_with_context = 0
    with FEEDBACK_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue          # 半行/脏行直接忽略，不要让统计接口 500
            total += 1
            if obj.get("feedback") == "good":
                good += 1
            elif obj.get("feedback") == "bad":
                bad += 1
                if obj.get("contexts"):
                    bad_with_context += 1
    return {"total": total, "good": good, "bad": bad, "bad_with_context": bad_with_context}


@app.post("/feedback", dependencies=[Depends(require_api_key)])
async def submit_feedback(req: FeedbackRequest):
    """接收前端点赞/踩，追加写入 feedback_log.jsonl。

    鉴权：与 /chat/stream 同一把钥匙（X-API-Key / Authorization: Bearer）。
    为什么不能匿名开放——这个接口会把 question + answer + contexts 原样落盘，
    匿名可达等于"任何人都能往你的数据飞轮里灌垃圾"，归因分析直接失真。

    请求示例：
        POST /feedback
        {
          "session_id": "uuid-xxx",
          "question": "RAG 的重排是怎么做的？",
          "answer": "……",
          "feedback": "bad",
          "contexts": ["【来源：ch3.md】\\n重排使用 bge-reranker……", "..."],
          "sources": ["ch3.md"],
          "confidence": 0.62,
          "message_id": "a1b2c3d4"
        }

    返回：
        {"ok": true, "feedback_id": "...", "saved_to": "feedback_log.jsonl"}
    """
    feedback_id = uuid.uuid4().hex
    contexts = _sanitize_contexts(req.contexts)

    record = {
        # ---- 核心字段（题目要求的 4 个）----
        "session_id": req.session_id or "default_session",
        "question": (req.question or "")[:MAX_QUESTION_CHARS],
        "answer": (req.answer or "")[:MAX_ANSWER_CHARS],
        "feedback": req.feedback,
        # ★ 证据字段：没有它，离线归因只能靠猜（全被判成"检索未命中"）
        "contexts": contexts,
        "contexts_count": len(contexts),
        # ---- 追踪字段：出问题时能对上号 ----
        "feedback_id": feedback_id,
        "message_id": req.message_id,
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        # ---- 可选上下文 ----
        "sources": req.sources or [],
        "confidence": req.confidence,
        "comment": (req.comment or "")[:MAX_COMMENT_CHARS] if req.comment else None,
    }

    try:
        # 文件 IO 放到线程池，避免阻塞事件循环（其他 SSE 连接不会被拖慢）
        await asyncio.to_thread(_append_feedback_line, record)
    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ 反馈写入失败: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"反馈写入失败: {e}")

    logger.info(
        f"📝 收到反馈[{req.feedback}] session={record['session_id'][:8]} "
        f"id={feedback_id[:8]} contexts={len(contexts)} q={record['question'][:30]!r}"
    )
    return {
        "ok": True,
        "feedback_id": feedback_id,
        "saved_to": FEEDBACK_LOG_PATH.name,
    }


@app.get("/feedback/stats", dependencies=[Depends(require_api_key)])
async def feedback_stats():
    """【附赠】看一眼当前反馈分布，验证前端到底有没有写进来。

    例：curl -H "X-API-Key: <密钥>" http://127.0.0.1:8000/feedback/stats
    统计结果会暴露"哪些问题被踩了"，同样不适合匿名开放。
    """
    try:
        stats = await asyncio.to_thread(_read_feedback_stats)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"统计失败: {e}")

    bad = stats.get("bad", 0)
    total = stats.get("total", 0)
    with_ctx = stats.get("bad_with_context", 0)
    return {
        **stats,
        "bad_rate": round(bad / total, 4) if total else 0.0,
        # 证据覆盖率：bad 里有多少条带 contexts。低于 100% 说明还有老数据/前端没升级。
        "bad_context_coverage": round(with_ctx / bad, 4) if bad else 0.0,
        "log_path": str(FEEDBACK_LOG_PATH),
    }


# 注意：本端点必须定义在 require_api_key 之后。
# 装饰器在导入期就会求值，放前面会直接 NameError，表现为容器反复重启。
@app.post("/admin/cache/clear", dependencies=[Depends(require_api_key)])
def clear_retrieval_cache():
    """清空检索缓存。

    为什么值得一个端点：评测要测的是**模型自身的方差**。缓存开着时，
    同一批问题重复跑拿到的是同一份缓存答案，方差恒为 0 —— 测了等于没测。
    评测脚本在每一轮重复前调这里，才能拿到真实的冷路径分布。

    线上也用得上：改了检索策略想立刻生效，不必重启后端（重启要重载 1.2GB 模型）。
    """
    cache = _retrieval_cache.get_cache()
    before = cache.stats()
    cache.clear()
    return {"cleared": True, "before": before, "after": cache.stats()}


# ================= 启动代码 =================
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 正在启动 FastAPI 服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
