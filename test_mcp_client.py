# test_mcp_client.py
import asyncio
from fastmcp import Client

async def main():
    # 1. 指向你刚写的 Server 文件
    async with Client("mcp_server.py") as client:
        # 2. 测试知识库检索工具
        result = await client.call_tool("knowledge_base_query", {"query": "小智喜欢什么？"})
        print("知识库查询结果:", result)

        # 3. 测试知识图谱工具
        result = await client.call_tool("knowledge_graph_query", {"query": "小陈住在哪个省？"})
        print("知识图谱查询结果:", result)

if __name__ == "__main__":
    asyncio.run(main())