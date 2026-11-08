#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_mcp_client.py —— 用 MCP 客户端调一下 mcp_server.py，确认工具可用。

注意（踩过的坑）
────────────────
直接写 `Client("mcp_server.py")` 在装了独立依赖目录的机器上会 `Connection closed`：
客户端起的子进程没有继承 PYTHONPATH，import fastmcp 直接失败就退出了。
这里显式用 StdioTransport 指定解释器 + 环境变量 + 工作目录，稳定得多。

用法
────
    PYTHONPATH=/root/mcp_lib python3 test_mcp_client.py
"""

import asyncio
import os
import sys

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

HERE = os.path.dirname(os.path.abspath(__file__))


async def main() -> None:
    env = dict(os.environ)

    transport = StdioTransport(
        command=sys.executable,
        args=[os.path.join(HERE, "mcp_server.py")],
        env=env,
        cwd=HERE,
    )

    async with Client(transport) as client:
        tools = await client.list_tools()
        print("可用工具：", [t.name for t in tools])

        for name, query in [
            ("knowledge_base_query", "小智喜欢什么？"),
            ("knowledge_graph_query", "小陈住在哪个省？"),
        ]:
            res = await client.call_tool(name, {"query": query})
            text = res.content[0].text if getattr(res, "content", None) else str(res)
            print(f"\n[{name}] {query}\n{text}")


if __name__ == "__main__":
    asyncio.run(main())
