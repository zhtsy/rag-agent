# -*- coding: utf-8 -*-
"""mcp_server.py 的单元测试。

这一层的测试真正要防的是三件事，而不是"函数能跑"：
  1. **工具真的注册进 MCP 注册表了吗**——定义了一个 async 函数但忘了 @server.tool()，
     或者工具名拼错，本地完全看不出问题，客户端只会说"没有可用工具"。
  2. **错误有没有被翻译成可执行的建议**——"HTTP 401"对调用方毫无用处，
     "请检查 RAG_API_KEY"才有用。
  3. **失败时是不是抛异常**——MCP 客户端拿到未捕获异常会整轮崩掉，
     而这本来只是一句"后端没起来"。

依赖官方 SDK 与 httpx，没装就整文件跳过（宿主机默认没有，
它们装在 .mcp_libs 里，见 OPS.md）。运行方式：

    PYTHONPATH=.mcp_libs python3 -m pytest tests/test_mcp_server.py -q
"""

import asyncio
import inspect

import pytest

pytest.importorskip("httpx", reason="需要 httpx（pip install --target .mcp_libs mcp httpx）")
# 必须检查**子模块**而不是顶层 `mcp`：环境里装着 1.x 时 `import mcp` 照样成功，
# 但 mcp_server.py 用的是 2.x 才有的 mcp.server.mcpserver，于是报错发生在
# **收集阶段**，pytest 会因 collection error 中断，整个测试套件一个都跑不了。
# 实测：/home/zh/myenv 里是 mcp 1.24.0，改这一行之前 pytest 直接 1 error 退出。
pytest.importorskip("mcp.server.mcpserver", reason="需要官方 MCP SDK 2.x，见 OPS.md 第 7 节")

import mcp_server as m


def run(coro):
    """asyncio.run 的短别名：这些工具都是 async，而项目没引入 pytest-asyncio。"""
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# 1. 契约：工具确实注册进了 MCP 层
# ----------------------------------------------------------------------
class TestRegistry:
    def test_exactly_two_tools_registered(self):
        """工具数量是设计决策，不是巧合。

        每多一个工具，模型选错的概率就上升一点；工具多到枚举不过来时，
        才需要退化成"循环探索"的 agent 范式。这里刻意只留两个动词。
        """
        tools = run(m.server.list_tools())
        names = {t.name for t in tools}
        assert names == {"search_knowledge_base", "ask_knowledge_base"}

    def test_tools_have_descriptions(self):
        """描述为空等于模型不知道该何时调用——比没有工具更糟。

        注意属性名是 `input_schema`（snake_case）：Python 侧用下划线，
        序列化到 JSON-RPC 线路上才转成 `inputSchema`。
        这里要的是**参数表非空**，也就是每个工具都声明了入参。
        """
        for tool in run(m.server.list_tools()):
            assert tool.description and len(tool.description) > 20, tool.name
            assert tool.input_schema.get("properties"), tool.name

    def test_server_metadata(self):
        assert m.server.name == "rag-knowledge-base"
        assert m.server.version == "1.0.0"


# ----------------------------------------------------------------------
# 2. 响应解释：状态码 → 可执行的建议
# ----------------------------------------------------------------------
class TestInterpretResponse:
    def test_ok_json(self):
        ok, data = m._interpret_response(200, '{"kind": "evidence", "hits": []}')
        assert ok is True
        assert data["kind"] == "evidence"

    def test_401_points_to_config(self):
        ok, msg = m._interpret_response(401, "")
        assert ok is False
        assert "RAG_API_KEY" in msg

    def test_503_says_retry_is_useless(self):
        """503 是服务端没配密钥，重试一万次也不会好——必须明说。"""
        ok, msg = m._interpret_response(503, "")
        assert ok is False
        assert "重试无用" in msg

    def test_429_tells_to_wait(self):
        ok, msg = m._interpret_response(429, "")
        assert ok is False
        assert "5 秒" in msg

    def test_invalid_json(self):
        ok, msg = m._interpret_response(200, "<html>502 Bad Gateway</html>")
        assert ok is False
        assert "JSON" in msg


# ----------------------------------------------------------------------
# 3. 配置读取
# ----------------------------------------------------------------------
class TestConfig:
    def test_base_url_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("RAG_BASE_URL", "http://example.com:8000/")
        assert m._base_url() == "http://example.com:8000"

    def test_base_url_default(self, monkeypatch):
        monkeypatch.delenv("RAG_BASE_URL", raising=False)
        assert m._base_url() == m.DEFAULT_BASE_URL

    def test_timeout_bad_value_falls_back(self, monkeypatch):
        """环境变量写成 "abc" 不能让整个服务起不来——配置错误要能降级。"""
        monkeypatch.setenv("RAG_TIMEOUT", "abc")
        assert m._timeout() == m.DEFAULT_TIMEOUT_S

    def test_api_key_env_wins(self, monkeypatch, tmp_path):
        f = tmp_path / "k"
        f.write_text("from-file\n", encoding="utf-8")
        monkeypatch.setenv("RAG_API_KEY", "from-env")
        monkeypatch.setenv("RAG_API_KEY_FILE", str(f))
        assert m._load_api_key() == "from-env"

    def test_api_key_falls_back_to_file(self, monkeypatch, tmp_path):
        """复用后端的密钥文件，避免"两处各配一份、改了一处忘另一处"。"""
        f = tmp_path / "k"
        f.write_text("  file-secret  \n", encoding="utf-8")
        monkeypatch.delenv("RAG_API_KEY", raising=False)
        monkeypatch.setenv("RAG_API_KEY_FILE", str(f))
        assert m._load_api_key() == "file-secret"

    def test_api_key_missing_is_empty_not_crash(self, monkeypatch, tmp_path):
        monkeypatch.delenv("RAG_API_KEY", raising=False)
        monkeypatch.setenv("RAG_API_KEY_FILE", str(tmp_path / "nope"))
        assert m._load_api_key() == ""


# ----------------------------------------------------------------------
# 4. 答案渲染：把内部 status 语义翻译给外部 agent
# ----------------------------------------------------------------------
class TestFormatAnswer:
    def test_basic(self):
        out = m._format_answer({
            "answer": "小智住在南京。",
            "sources": ["test.txt"],
            "confidence": 0.9994,
            "status": "answered",
        })
        assert "小智住在南京。" in out
        assert "test.txt" in out
        assert "0.9994" in out

    def test_confidence_labelled_as_relevance_not_accuracy(self):
        """必须写成"证据相关度"。

        写成"置信度 99%"会被调用方当成"答对概率 99%"，
        然后拿它去做业务判断——而这个数只是 rerank 分，没有那个含义。
        """
        out = m._format_answer({"answer": "x", "confidence": 0.87, "status": "answered"})
        assert "相关度" in out
        assert "置信度" not in out

    @pytest.mark.parametrize("status,keyword", [
        ("out_of_scope", "正常结果"),
        ("missed", "服务侧问题"),
        ("ungrounded", "低可信"),
    ])
    def test_status_translated(self, status, keyword):
        assert keyword in m._format_answer({"answer": "x", "status": status})

    def test_answered_has_no_warning(self):
        out = m._format_answer({"answer": "x", "status": "answered"})
        assert "低可信" not in out and "服务侧问题" not in out

    def test_empty_answer(self):
        assert "没有产出" in m._format_answer({"answer": "", "status": "answered"})

    def test_null_confidence_omitted(self):
        """拒答时 confidence 是 null，不能渲染成 0.0000（那会被读成"答得很差"）。"""
        out = m._format_answer({"answer": "知识库无相关内容", "confidence": None,
                                "status": "out_of_scope"})
        assert "相关度" not in out


# ----------------------------------------------------------------------
# 5. 工具行为
# ----------------------------------------------------------------------
class TestSearchTool:
    def test_formats_evidence(self, monkeypatch):
        async def fake(path, payload):
            assert path == "/search"
            assert payload == {"query": "小智住哪", "top_k": 3}
            return True, {
                "kind": "evidence",
                "hits": [{"source": "a.md", "score": 0.9, "content": "小智住在南京。",
                          "kind": "vector"}],
                "elapsed_ms": 120.0, "cached": False,
            }

        monkeypatch.setattr(m, "_post_json", fake)
        out = run(m.search_knowledge_base("小智住哪", top_k=3))
        assert "a.md" in out and "0.9000" in out and "小智住在南京" in out
        assert "未命中缓存" in out

    def test_top_k_is_clamped(self, monkeypatch):
        """top_k=999 要收敛到 20，不能原样透传给后端（省得它被拿来打库）。"""
        async def fake(path, payload):
            assert payload["top_k"] == 20
            return True, {"kind": "no_match", "hits": [], "message": "", "raw": "",
                          "elapsed_ms": 1.0, "cached": False}

        monkeypatch.setattr(m, "_post_json", fake)
        run(m.search_knowledge_base("x", top_k=999))

    def test_empty_query_short_circuits(self, monkeypatch):
        """空 query 不该发请求——省一次往返，也避免后端白跑一次检索。"""
        called = {"n": 0}

        async def fake(path, payload):
            called["n"] += 1
            return True, {}

        monkeypatch.setattr(m, "_post_json", fake)
        out = run(m.search_knowledge_base("   "))
        assert called["n"] == 0
        assert "不能为空" in out

    def test_backend_failure_is_text_not_exception(self, monkeypatch):
        """后端挂了要返回一句话，而不是抛异常把客户端整轮打断。"""
        async def fake(path, payload):
            return False, "无法连接知识库服务（http://127.0.0.1:8000）。"

        monkeypatch.setattr(m, "_post_json", fake)
        out = run(m.search_knowledge_base("x"))
        assert "无法连接" in out


class TestAskTool:
    def test_aggregates_answer(self, monkeypatch):
        async def fake(payload):
            assert payload["message"] == "小智住在哪"
            assert payload["session_id"] == "s1"
            return True, {"answer": "南京。", "sources": ["test.txt"],
                          "confidence": 0.9994, "status": "answered"}

        monkeypatch.setattr(m, "_stream_answer", fake)
        out = run(m.ask_knowledge_base("小智住在哪", session_id="s1"))
        assert "南京。" in out and "test.txt" in out

    def test_default_session(self, monkeypatch):
        seen = {}

        async def fake(payload):
            seen.update(payload)
            return True, {"answer": "x", "status": "answered"}

        monkeypatch.setattr(m, "_stream_answer", fake)
        run(m.ask_knowledge_base("q"))
        assert seen["session_id"] == "mcp"

    def test_empty_question(self, monkeypatch):
        async def fake(payload):
            raise AssertionError("不该发请求")

        monkeypatch.setattr(m, "_stream_answer", fake)
        assert "不能为空" in run(m.ask_knowledge_base("  "))

    def test_failure_passthrough(self, monkeypatch):
        async def fake(payload):
            return False, "生成超时（>180s）。"

        monkeypatch.setattr(m, "_stream_answer", fake)
        assert "生成超时" in run(m.ask_knowledge_base("q"))
