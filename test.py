# import ollama
# import streamlit
# print(f"Ollama当前可用的模型有：{ollama.list()}")
# print(f"streamlit库的版本是：{streamlit.__version__}")
import ollama
import streamlit
# 获得ollama的客户端对象
client = ollama.Client()

try:
    # 列出有哪些模型可用
    models = client.list()
    print("✅连接成功，模型列表：", models)
except Exception as e:
    print("❌错误信息：", e)
print(f"streamlit库的版本是：{streamlit.__version__}")
# 展示模型的详细信息
print(client.show("deepseek-r1:7b"))
# 列出有哪些模型在运行中
print(client.ps())

while True:
    prompt = input("请输入问题：")
    # .chat和模型进行对话
    response = client.chat(
        model = "deepseek-r1:7b",
        messages = [{"role" : "user","content" : prompt}]
)

    print(response["message"]["content"])

# import os
# import ollama

# # 1. 看看 PyCharm 到底读到环境变量没
# print("PyCharm 读到的 OLLAMA_HOST:", os.environ.get("OLLAMA_HOST"))
#
# # 2. 创建一个不传参的客户端，看它默认连到哪里
# client_default = ollama.Client()
# print("默认 client 连的地址:", client_default._client._base_url)
#
# # 3. 尝试强制硬编码你设置的那个 IP
# client_hard = ollama.Client(host="http://172.20.176.1:11434")
# print("硬编码 client 连的地址:", client_hard._client._base_url)
#
# # 4. 分别打印 ps
# print("默认客户端的 ps:", client_default.ps())
# print("硬编码客户端的 ps:", client_hard.ps())