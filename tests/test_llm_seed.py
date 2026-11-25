"""确认生成侧钉死了 seed。

背景：temperature=0 不等于确定性。实测 Ollama 不带 seed 时每次随机取种子，
同一 prompt 连发 3 次得到 3 种输出；补上 seed 后才逐字一致。评测里这个差异
表现为「同一上下文、同一题，1 过 2 挂」，很容易被误判成不可复现的抖动。
"""

import os

import rag_agent


def test_seed_is_int_and_configurable():
    assert isinstance(rag_agent.LLM_SEED, int)
    # 默认 42；可用 RAG_LLM_SEED 覆盖，方便做「换种子 robustness」实验
    if not os.getenv("RAG_LLM_SEED"):
        assert rag_agent.LLM_SEED == 42


def test_cloud_llm_carries_seed():
    assert rag_agent.cloud_llm.temperature == 0
    assert rag_agent.cloud_llm.seed == rag_agent.LLM_SEED


def test_local_llm_carries_seed():
    """hybrid 时是 ChatOllama，全 DeepSeek 时是 ChatDeepSeek，两条路都得带 seed。"""
    assert rag_agent.local_llm.temperature == 0
    assert rag_agent.local_llm.seed == rag_agent.LLM_SEED
    assert rag_agent.local_llm_stream.seed == rag_agent.LLM_SEED
