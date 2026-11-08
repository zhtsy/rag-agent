# -*- coding: utf-8 -*-
"""mcp_server.py —— 把本地 RAG 知识库暴露成 MCP 工具。

为什么是「薄客户端」而不是「在本地再跑一套检索」
------------------------------------------------
这台机器只有 2 vCPU / 3.6G 内存，backend 已经常驻了 bge 向量模型 + reranker
（约 1.2G）。MCP Server 如果自己 import 一遍 rag_agent，会再加载一份模型，
直接把机器打爆——而且两份模型的检索结果还可能因为缓存不同而不一致。

所以这里的职责只有三件：
  1. 把 MCP 的工具调用翻译成 HTTP 请求；
  2. 把后端返回的结构化证据翻译成模型好读的文本；
  3. 把各种失败翻译成**能指导下一步动作**的话（重试 / 别重试 / 改配置）。

真正的检索、排序、缓存、鉴权都在 backend 里，一行都没有重复实现。

为什么只暴露两个工具
--------------------
工具越多，模型选错的概率越高，越需要"循环探索"的 agent 范式来兜底。
这里只给两个界限清楚的动词：
  · search —— 只要证据，不要答案（快，省一次 LLM 调用）
  · ask    —— 要一个组织好的答案（慢，但可直接转述给用户）
够用了。加第三个（比如"列出所有文档"）只会稀释选择准确率。

用法
----
    # stdio（推荐，客户端自己拉起进程）
    PYTHONPATH=.mcp_libs python3 mcp_server.py

    # SSE（常驻服务，多人共用一台机器时）
    PYTHONPATH=.mcp_libs python3 mcp_server.py --transport sse --port 8765

环境变量：
    RAG_BASE_URL     后端地址，默认 http://127.0.0.1:8000
    RAG_API_KEY      访问密钥；不设则读 RAG_API_KEY_FILE 或同目录 .rag_api_key
    RAG_TIMEOUT      单次请求超时秒数，默认 180
    MCP_TRANSPORT    stdio | sse | streamable-http
    MCP_PORT         SSE 端口，默认 8765
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from mcp.server.mcpserver import MCPServer

from evidence import format_evidence

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_S = 180.0

# ---- 状态语义 → 给调用方的提醒 ----------------------------------------
# 这套 status 是后端为防止"拒答被当成故障、硬答被当成正常"而设计的，
# 外部 agent 看不见这层语义就会误判，所以在这里翻译成人话。
_STATUS_NOTES = {
    "out_of_scope": "（知识库未覆盖这个话题，拒答是正常结果，不是故障）",
    "missed": "（后端检索到了证据却没作答，属于服务侧问题，可重试或反馈）",
    "ungrounded": "（本次回答缺少检索依据，属于低可信输出，转述时请标注不确定）",
}


def _base_url() -> str:
    """每次调用都读环境变量，不在导入期固化——便于测试注入和运行时切换。"""
    return (os.environ.get("RAG_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _timeout() -> float:
    try:
        return float(os.environ.get("RAG_TIMEOUT", DEFAULT_TIMEOUT_S))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _load_api_key() -> str:
    """密钥优先取环境变量，其次读文件。

    读文件而不是只认环境变量，是因为后端的密钥就存在 `.rag_api_key` 里
    （轮换时不重启服务）。MCP 直接复用同一个来源，省掉"两处各配一份、
    改了一处忘了另一处"的经典事故。
    """
    key = (os.environ.get("RAG_API_KEY") or "").strip()
    if key:
        return key
    path = Path(
        os.environ.get("RAG_API_KEY_FILE")
        or (Path(__file__).resolve().parent / ".rag_api_key")
    )
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = _load_api_key()
    if key:
        headers["X-API-Key"] = key
    return headers


def _status_error(code: int, body: str) -> str:
    """把 HTTP 状态码翻译成"下一步该做什么"。

    只报 "HTTP 401" 对调用方没有用——它不知道该改配置、该重试、还是该换个问题。
    """
    if code == 401:
        return ("访问密钥无效或缺失（401）。请检查 RAG_API_KEY 环境变量，"
                "或 mcp_server.py 同目录下的 .rag_api_key 文件。")
    if code == 503:
        return ("后端未配置访问密钥（503）。这是服务端配置问题，重试无用，"
                "需要在服务器上检查 .rag_api_key 是否存在且非空。")
    if code == 429:
        return "知识库服务繁忙（429）。等约 5 秒再试；持续繁忙说明并发额度需要调大。"
    return f"知识库服务返回 {code}：{body[:300]}"


def _interpret_response(status_code: int, text: str) -> Tuple[bool, Any]:
    """把 HTTP 响应解释成 (ok, data 或 错误文本)。

    单独抽出来是为了能单测——错误映射是这一层最容易写错、
    又最不容易在"一切正常"的演示里被发现的部分。
    """
    if status_code >= 400:
        return False, _status_error(status_code, text)
    try:
        return True, json.loads(text)
    except ValueError:
        return False, "知识库服务返回的不是合法 JSON（响应体可能被代理改写）。"


async def _post_json(path: str, payload: Dict[str, Any]) -> Tuple[bool, Any]:
    """POST 一个 JSON，返回 (ok, data 或 错误文本)。

    **任何异常都不向上抛**：MCP 客户端拿到未捕获异常的表现是整轮对话崩掉，
    而这里失败明明可以用一句"服务不可达，请检查后端"交代清楚。
    """
    url = f"{_base_url()}{path}"
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            resp = await client.post(url, json=payload, headers=_headers())
    except httpx.ConnectError:
        return False, (f"无法连接知识库服务（{url}）。请确认后端已启动："
                       f"docker compose ps 里 backend 应为 healthy。")
    except httpx.TimeoutException:
        return False, (f"知识库服务响应超时（>{_timeout():.0f}s）。"
                       f"可能是首次请求正在加载模型，或机器负载过高。")
    except httpx.HTTPError as e:
        return False, f"请求知识库服务失败：{type(e).__name__}: {e}"

    return _interpret_response(resp.status_code, resp.text)


async def _stream_answer(payload: Dict[str, Any]) -> Tuple[bool, Any]:
    """消费后端的 SSE，聚合成一条完整答案。

    MCP 的工具调用是"一问一答"，没有增量渲染的语义，所以流式在这里没有收益——
    直接读到底再返回，客户端拿到的是完整结果，不会看到半截回答。
    """
    url = f"{_base_url()}/chat/stream"
    pieces: List[str] = []
    meta: Dict[str, Any] = {}

    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", url, json=payload, headers=_headers()) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    return False, _status_error(resp.status_code, body)

                async for line in resp.aiter_lines():
                    # SSE 里 `: connected` 是注释帧（后端用来提前顶出响应头），跳过
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if "text" in obj:
                        pieces.append(obj["text"])
                    elif obj.get("error"):
                        return False, f"生成过程中出错：{obj['error']}"
                    else:
                        meta = obj          # 收尾帧：sources / confidence / status / contexts
    except httpx.ConnectError:
        return False, (f"无法连接知识库服务（{_base_url()}）。请确认后端已启动。")
    except httpx.TimeoutException:
        return False, f"生成超时（>{_timeout():.0f}s）。可以先用 search 看看检索是否正常。"

    return True, {"answer": "".join(pieces).strip(), **meta}


def _format_answer(data: Dict[str, Any]) -> str:
    """把后端收尾帧渲染成可转述的文本。"""
    answer = (data.get("answer") or "").strip()
    lines: List[str] = [answer if answer else "（模型没有产出任何内容）"]

    sources = [s for s in (data.get("sources") or []) if s]
    if sources:
        lines.append("来源：" + "、".join(str(s) for s in sources))

    conf = data.get("confidence")
    if isinstance(conf, (int, float)):
        # confidence 的含义是"最相关那条证据的 rerank 分"，不是"答对的概率"。
        # 这里写成"证据相关度"，避免调用方把它当准确率去用。
        lines.append(f"最强证据相关度：{conf:.4f}")

    note = _STATUS_NOTES.get(str(data.get("status") or ""))
    if note:
        lines.append(note)

    return "\n".join(lines)


# ====================== MCP Server ======================
server = MCPServer(
    name="rag-knowledge-base",
    title="本地 RAG 知识库",
    instructions=(
        "一个基于本地知识库的检索与问答服务，包含向量检索、BM25、知识图谱三路召回和重排。\n"
        "两个工具的分工：\n"
        "  · search_knowledge_base —— 只返回证据片段（快，适合自己推理、比对、引用）；\n"
        "  · ask_knowledge_base   —— 返回组织好的答案（慢，适合直接转述给用户）。\n"
        "重要：如果 search 返回「知识库中没有相关内容」，说明知识库确实没有覆盖这个话题，"
        "换一种问法通常也不会命中，请如实告知用户，不要反复尝试或凭常识编造。"
    ),
    version="1.0.0",
)


@server.tool()
async def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """在本地知识库中检索与问题相关的证据片段（只检索，不生成答案）。

    适用场景：需要先确认知识库里有没有相关资料，或要基于原始证据自己推理、比对、引用。

    Args:
        query: 检索句子，必须自包含。含"他/她/它"时请先替换成具体实体名——
            检索层不做指代消解，带代词查不到东西。
        top_k: 最多返回几条证据（1~20，默认 5）。只能往下截：
            上游选块数量由检索策略决定，要 20 条也只给得到实际选出来的那几条。
    """
    q = (query or "").strip()
    if not q:
        return "检索失败：query 不能为空。"

    try:
        k = max(1, min(int(top_k), 20))
    except (TypeError, ValueError):
        k = 5

    ok, data = await _post_json("/search", {"query": q, "top_k": k})
    if not ok:
        return data
    return format_evidence(
        data,
        elapsed_ms=data.get("elapsed_ms"),
        cached=data.get("cached"),
    )


@server.tool()
async def ask_knowledge_base(question: str, session_id: str = "mcp") -> str:
    """向本地知识库提问，返回组织好的答案、引用来源和证据相关度。

    适用场景：要把结果直接转述给用户，不想自己再组织一遍语言。

    Args:
        question: 完整的问题。可以带上下文，但同一话题请用同一个 session_id。
        session_id: 会话标识（默认 "mcp"）。同一个 session 内多轮调用会共享上下文，
            代词可以指代前文提到的实体；**换话题时请换一个 session_id**，
            否则历史里的实体可能被误带进新问题。
    """
    q = (question or "").strip()
    if not q:
        return "提问失败：question 不能为空。"

    ok, data = await _stream_answer({
        "message": q,
        "session_id": (session_id or "mcp").strip() or "mcp",
    })
    if not ok:
        return data
    return _format_answer(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 RAG 知识库的 MCP 服务")
    parser.add_argument(
        "--transport",
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse", "streamable-http"],
        help="传输方式（默认 stdio，由客户端拉起进程）",
    )
    parser.add_argument(
        "--host", default=os.environ.get("MCP_HOST", "127.0.0.1"),
        help="SSE/HTTP 模式监听地址（默认只监听本机）",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("MCP_PORT", "8765")),
        help="SSE/HTTP 模式端口",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        # stdio 下 stdout 是协议通道，任何 print 都会污染 JSON-RPC 流。
        # 所以这个脚本里**一个 print 都不许有**，要输出只能走 stderr。
        server.run("stdio")
    else:
        server.run(args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
