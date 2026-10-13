from typing import Literal

from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from rich import print as rprint

model = init_chat_model("ollama:qwen2.5:7b")

@tool
def get_weather(query:str) ->str:
    """查询城市天气"""
    if "南京" in query.lower():
        return f"南京的气温为25度，多云转晴。"
    elif "北京" in query.lower():
        return f"北京的气温为26度，阴天。"
    elif "上海" in query.lower():
        return f"上海的气温为27度，下雨。"
    elif "苏州" in query.lower():
        return f"苏州的气温为28度，晴天。"
    else:
        return f"{query}的天气暂时无法查询。"

@tool
def add(a:int,b:int) ->int:
    """计算两个数之和"""
    return a+b

@tool
def order_ticket(money:int) ->int:
    """根据金额选择预定相应机票"""
    if money >= 10000:
        return f"将预定头等舱。"
    elif 500<= money < 10000:
        return f"将预定普通舱。"
    elif 200<= money <500 :
        return f"将预定早鸟票。"
    else:
        return f"预算不足，无法购票。"

agent = create_agent(
    model = model,
    system_prompt="""请根据用户的需求来选择相应的工具""",
    tools = [get_weather,add,order_ticket]
)

result = agent.invoke({
    "messages": [HumanMessage(content="我想查询南京的天气"),
                 HumanMessage(content="我有10000块钱，帮我预订机票。")]
})

rprint(result)
