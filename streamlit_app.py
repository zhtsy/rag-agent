import streamlit as st
import time
import uuid
import random
from dotenv import load_dotenv
load_dotenv()

# 导入你的 Agent
from rag_agent import run_conversation


# ========== 页面配置 ==========
st.set_page_config(
    page_title="RAG 智能助手",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
    <style>
    .main-header { font-size: 2rem; font-weight: 600; color: #FF6600; text-align: center; margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🤖 RAG 智能助手</div>", unsafe_allow_html=True)


# ========== 会话状态初始化 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# ========== 侧边栏 ==========
with st.sidebar:
    st.header("设置")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.caption(f"会话 ID: {st.session_state.session_id[:8]}")


# ========== 显示历史消息 ==========
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ========== 打字机动画函数 ==========
def typewriter(text: str, placeholder, delay: float = 0.03):
    """逐字显示文本，模拟打字机效果"""
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(displayed + "▌")   # 光标效果
        time.sleep(delay)
    placeholder.markdown(displayed)   # 去掉光标，显示完整答案


# ========== 聊天输入 ==========
if prompt := st.chat_input("输入您的问题..."):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Loading 提示 + 调用 Agent
    with st.chat_message("assistant"):
        # ===== Loading 阶段 =====
        with st.spinner("🔍 正在检索知识库，请稍候..."):
            try:
                result = run_conversation(
                    prompt,
                    thread_id=st.session_state.session_id
                )
                # 提取答案
                if hasattr(result, "answer"):
                    answer = result.answer
                else:
                    answer = str(result)
            except Exception as e:
                answer = f"抱歉，处理失败：{str(e)}"

        # ===== 打字机阶段 =====
        placeholder = st.empty()
        typewriter(answer, placeholder, delay=0.03)

    # 3. 保存回答到历史
    st.session_state.messages.append({"role": "assistant", "content": answer})