import streamlit as st
import time
if "count" not in st.session_state:
    st.session_state["count"] = 1
st.title("测试标题")

st.divider()

prompt = st.chat_input("请输入你的问题：")

if prompt:
    st.chat_message("user").markdown(prompt)
    with st.spinner("思考中"):
        time.sleep(1)
        st.chat_message("assistant").markdown(f"我不会:{st.session_state['count']}")
        st.session_state["count"] = st.session_state["count"]+1

