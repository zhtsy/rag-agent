from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool


@tool
def get_weather(city:str):
    """获取城市的天气"""
    return(f"{city}天气晴朗，气温15℃")


model = init_chat_model(model="ollama:qwen2.5:7b")

agent = create_agent(
    model = model,
    tools = [get_weather]
)

result = agent.invoke({
    "messages" : [SystemMessage(content="你是一个查询天气的气候助手，如果用户问的问题与天气无关，你可以回答不知道。"),
                HumanMessage(content="请问今天北京的天气是什么")]
}

)
response = result["messages"][-1]
print(response)