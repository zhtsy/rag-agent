import streamlit as st
import time
# 获得标题
st.title("测试标题")
# write方法，可以在网页中渲染你提供的内容
st.write("小平，你好！")
#分隔符
st.divider()
# 聊天输入框
name = st.chat_input("请输入你的名字")
if name:
    st.write(f"你好，{name}")

# 等待提示框
with st.spinner("思考中"):
    time.sleep(5)
    st.write("思考完成")

# 消息容器
# 角色支持：user、assistan、ai、human
st.chat_message("user").markdown("你是谁")
st.chat_message("assistant").markdown("我是聊天机器人")