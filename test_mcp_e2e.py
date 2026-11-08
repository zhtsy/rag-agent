# -*- coding: utf-8 -*-
"""MCP 端到端验证：显式用 StdioTransport 起 server（保证继承 PYTHONPATH），
再用 MCP 客户端调工具。结果 base64 回传，避开终端中文转码。"""
import asyncio
import base64
import os
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


def b64(s: str) -> str:
    return base64.b64encode(str(s).encode("utf-8")).decode()


async def main():
    env = dict(os.environ)
    root = Path(__file__).resolve().parent  # 原来写死 /root/rag_deploy，本地跑必挂
    # MCP SDK 云端装在 /root/mcp_lib，本地按项目内 .mcp_libs 找（见 OPS.md 第 7 节）
    for cand in (root / ".mcp_libs", Path("/root/mcp_lib")):
        if cand.exists():
            env["PYTHONPATH"] = str(cand)
            break
    transport = StdioTransport(
        command=sys.executable,
        args=[str(root / "mcp_server.py")],
        env=env,
        cwd=str(root),
    )
    async with Client(transport) as client:
        tools = await client.list_tools()
        print("TOOLS:", [t.name for t in tools])

        for name, q in [
            ("knowledge_base_query", "小智喜欢什么？"),
            ("knowledge_graph_query", "小陈住在哪个省？"),
        ]:
            res = await client.call_tool(name, {"query": q})
            text = res.content[0].text if getattr(res, "content", None) else str(res)
            print(f"{name} | Q_B64:{b64(q)}")
            print(f"{name} | A_B64:{b64(text)}")


if __name__ == "__main__":
    asyncio.run(main())
