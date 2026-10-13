import streamlit as st
import ollama

client = ollama.Client()

if "message" not in st.session_state:
    st.session_state["message"] = []

st.title("zh聊天机器人")

st.divider()

prompt = st.chat_input("请输入问题：")

if prompt:
    st.session_state["message"].append({"role":"user","content":prompt})

    for message in st.session_state["message"]:
        st.chat_message(message["role"]).markdown(message["content"])

    with st.spinner("AI思考中..."):
        response = client.chat(
            model = "deepseek-r1:7b",
            messages = [{"role":"user","content":prompt}]

        )
    st.session_state["message"].append({"role":"assistant","content":response["message"]["content"]})
    st.chat_message("assistant").markdown(response["message"]["content"])




