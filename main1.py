import re
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
import traceback

# 导入你的 Agent 代码（注意：确保路径正确）
# 假设你的 rag_agent.py 在同一个目录
from rag_agent import run_conversation, AgentAnswer

# 创建 FastAPI 应用
app = FastAPI(
    title="本地 RAG Agent API",
    description="基于 LangGraph + Qwen2.5 的本地知识库问答系统",
    version="1.0.0"
)


# 定义请求体结构
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"


# 定义响应体结构
class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    session_id: str


@app.get("/")
def root():
    return {"message": "RAG Agent API is running", "status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    与 Agent 对话的接口

    - **message**: 用户的问题
    - **session_id**: 会话ID（可选，用于多轮对话记忆）
    """
    try:
        # 调用你已有的 run_conversation 函数
        result = run_conversation(
            request.message,
            thread_id=request.session_id
        )

        # 处理返回结果
        if isinstance(result, AgentAnswer):
            return ChatResponse(
                answer=result.answer,
                sources=result.sources,
                confidence=result.confidence,
                session_id=request.session_id
            )
        else:
            # 如果返回的是字符串（解析失败的情况）
            return ChatResponse(
                answer=str(result),
                sources=[],
                confidence=0.0,
                session_id=request.session_id
            )
    except Exception as e:
        print("=" * 50)
        print("❌ 请求处理失败:")
        traceback.print_exc()
        print("=" * 50)
        raise HTTPException(status_code=500, detail=str(e))


# 健康检查端点
@app.get("/health")
def health():
    return {"status": "ok"}