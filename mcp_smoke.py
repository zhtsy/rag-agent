# -*- coding: utf-8 -*-
"""mcp_smoke.py —— 真实 MCP 协议握手验证。

为什么单元测试不够
------------------
单测能证明"函数逻辑对"，但证明不了"它是个 MCP Server"。
常见翻车方式单测一个都抓不到：
  · 忘了 @server.tool() → 工具没注册，客户端看到"没有可用工具"
  · stdout 里混进一个 print → JSON-RPC 流被污染，客户端直接断开
  · 参数 schema 生成失败 → 客户端不知道要传什么
  · initialize 协商失败 → 连不上，根本没有后续

所以这个脚本不 import mcp_server，而是**像真正的客户端那样**：
起子进程、用 stdio 发 JSON-RPC 帧、按 id 收响应。
它验证的是协议层，不是 Python 层。

跑法：
    PYTHONPATH=.mcp_libs python3 mcp_smoke.py

退出码 0 = 全通；非 0 = 有检查项失败。
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL_VERSION = "2024-11-05"      # 客户端提议版本；服务端可回自己支持的版本

_failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" —— {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


class StdioClient:
    """最小可用的 MCP stdio 客户端。

    ⚠️ 读 stdout 必须用**独立线程 + 队列**，不能用 select()。
    踩过的坑：`Popen(text=True, bufsize=1)` 会在 Python 层做缓冲，
    而 select() 检查的是 OS 层的 fd——数据已经被 Python 读进缓冲区，
    OS 那边就是"没有新数据"。于是 select 永远返回空，等到超时，
    表现为"服务端明明回了，客户端却一直挂着"。这个假死排查了很久，
    而单跑 `python3 mcp_server.py < init.json` 一直是正常的。
    """

    def __init__(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = ".mcp_libs" + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        # stdout 必须是干净的协议通道，所以进程的 stderr 单独收，失败时打印
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "mcp_server.py"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(HERE), env=env,
        )
        self._buf = ""
        self._queue: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """把 stdout 逐行搬进队列，直到 EOF。"""
        assert self.proc.stdout
        for line in self.proc.stdout:
            self._queue.put(line)
        self._queue.put(None)          # EOF 哨兵

    def send(self, obj: dict) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _read_line(self, timeout: float):
        try:
            return self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def recv(self, timeout: float = 120.0):
        """读到一条**可解析的** JSON-RPC 消息为止。

        解析失败的行直接丢弃并记录——如果服务端往 stdout 打了非协议内容，
        真正的客户端会断连，这里要能暴露出来。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line(deadline - time.time())
            if line is None:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                self._buf += line
                continue
        return None

    def request(self, method: str, params=None, msg_id=None, timeout: float = 120.0):
        req = {"jsonrpc": "2.0", "method": method}
        if msg_id is not None:
            req["id"] = msg_id
        if params is not None:
            req["params"] = params
        self.send(req)
        if msg_id is None:
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.recv(deadline - time.time())
            if msg is None:
                return None
            if msg.get("id") == msg_id:
                return msg
            # 服务端主动发的通知/日志：跳过，不打断等待
        return None

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()

    def stderr_text(self) -> str:
        try:
            return (self.proc.stderr.read() or "")[-1500:] if self.proc.stderr else ""
        except Exception:  # noqa: BLE001
            return ""


def text_of(result: dict) -> str:
    """把 tools/call 的 result 拍平成文本，便于断言。"""
    content = (result or {}).get("content") or []
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts)


def main() -> int:
    # 输出重定向到文件时 Python 默认是块缓冲：一旦卡住被 kill，进度会全部丢失，
    # 剩下的只有"日志为空 + 退出码 124"，什么也诊断不了。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass

    client = StdioClient()
    try:
        print("\n== 1. initialize ==")
        resp = client.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mcp-smoke", "version": "0.1.0"},
        }, msg_id=1)
        if not resp or "result" not in resp:
            check("initialize 成功", False, str(resp))
            print(client.stderr_text())
            return 1
        info = resp["result"].get("serverInfo") or {}
        check("initialize 成功",
              info.get("name") == "rag-knowledge-base",
              f"server={info.get('name')} v{info.get('version')} "
              f"protocol={resp['result'].get('protocolVersion')}")
        check("声明 tools 能力", "tools" in (resp["result"].get("capabilities") or {}))

        # 协议要求：初始化完成后必须补一条 initialized 通知
        client.request("notifications/initialized", msg_id=None)
        time.sleep(0.2)

        print("\n== 2. tools/list ==")
        resp = client.request("tools/list", msg_id=2, timeout=30)
        tools = (resp or {}).get("result", {}).get("tools", [])
        names = sorted(t.get("name") for t in tools)
        check("工具列表正确", names == ["ask_knowledge_base", "search_knowledge_base"],
              str(names))
        for t in tools:
            schema = t.get("inputSchema") or {}
            props = sorted((schema.get("properties") or {}).keys())
            check(f"  {t['name']} 有 schema",
                  bool(props) and bool(t.get("description")),
                  f"参数={props}")

        print("\n== 3. tools/call search_knowledge_base ==")
        began = time.time()
        resp = client.request("tools/call", {
            "name": "search_knowledge_base",
            "arguments": {"query": "小智住在哪个城市", "top_k": 2},
        }, msg_id=3, timeout=180)
        text = text_of((resp or {}).get("result", {}))
        cost = time.time() - began
        check("search 返回内容", bool(text.strip()) and not (resp or {}).get("error"),
              f"{cost:.1f}s")
        check("search 带出处", ("来源" in text), text.splitlines()[0] if text else "")
        print("     ── 返回摘要 ──")
        for line in text.strip().splitlines()[:6]:
            print(f"     {line}")
        if (resp or {}).get("error"):
            print("     error:", resp["error"])

        print("\n== 4. tools/call ask_knowledge_base ==")
        began = time.time()
        resp = client.request("tools/call", {
            "name": "ask_knowledge_base",
            "arguments": {"question": "小智住在哪个城市？", "session_id": "mcp-smoke"},
        }, msg_id=4, timeout=240)
        text = text_of((resp or {}).get("result", {}))
        cost = time.time() - began
        check("ask 返回答案", bool(text.strip()) and not (resp or {}).get("error"),
              f"{cost:.1f}s")
        check("ask 带来源", "来源" in text)
        print("     ── 返回摘要 ──")
        for line in text.strip().splitlines()[:6]:
            print(f"     {line}")
        if (resp or {}).get("error"):
            print("     error:", resp["error"])

        # 服务端如果往 stdout 混了非协议内容，这里会抓到
        check("stdout 未被污染", client._buf == "",
              client._buf[:200] if client._buf else "")

    finally:
        # 必须先关进程再读 stderr。
        # 反过来会死锁：进程还活着时 stderr 管道没有 EOF，read() 会一直等，
        # 表现为"所有检查都 PASS，但脚本永远不退出"（退出码 124）。
        client.close()
        err = client.stderr_text()

    print()
    if _failures:
        print(f"❌ {len(_failures)} 项失败: {_failures}")
        if err:
            print("--- 服务端 stderr ---")
            print(err)
        return 1
    print("✅ MCP 协议握手全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
