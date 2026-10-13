import streamlit as st
import time
if "count" not in st.session_state:
    st.session_state["count"] = 1
if "message" not in st.session_state:
    st.session_state["message"] = []
st.title("测试标题")

st.divider()

prompt = st.chat_input("请输入你的问题：")

if prompt:
    st.session_state["message"].append({"role":"user","content":prompt})
    for message in st.session_state["message"]:
        st.chat_message(message["role"]).markdown(message["content"])

    with st.spinner("思考中..."):
        time.sleep(1)
        response = f"我不会：{st.session_state["count"]}"
        st.session_state["message"].append({"role":"assistant","content":response})
        st.session_state["count"] = st.session_state["count"] + 1
        st.chat_message("assistant").markdown(response)

