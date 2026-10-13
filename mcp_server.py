# mcp_server.py
import sys
# ========== 关键：MCP 协议要求 stdout 只输出 JSON-RPC ==========
# 把当前 stdout 保存起来，然后把默认输出重定向到 stderr
_real_stdout = sys.stdout
sys.stdout = sys.stderr

from rag_agent import search_knowledge_base, search_knowledge_graph

sys.stdout = _real_stdout
from fastmcp import FastMCP
# 1. 创建 MCP Server 实例
mcp = FastMCP("MyRAGTools")

# 2. 用 @mcp.tool 装饰器注册你的检索工具
@mcp.tool
def knowledge_base_query(query: str) -> str:
    """
    从本地知识库中检索信息，用于回答关于人物、美食、游戏等事实性问题。
    """
    # 直接调用你原来的 search_knowledge_base 函数
    return search_knowledge_base.invoke(query)

@mcp.tool
def knowledge_graph_query(query: str) -> str:
    """
    查询知识图谱，用于处理需要多跳推理的关系型问题，例如“谁住在哪个省”。
    """
    # 直接调用你原来的 search_knowledge_graph 函数
    return search_knowledge_graph.invoke(query)

# 3. 启动服务器
if __name__ == "__main__":
    # 使用 stdio 传输模式，这是本地客户端的标准连接方式
    mcp.run(transport="stdio")