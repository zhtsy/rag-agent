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
def search_database(query:str) -> str:
    """
    在客户数据库中搜索信息
    :param query:
    :return:
    """

    if "张三" in query.lower():
        return f"客户记录：张三，VIP客户，最近消费记录：2026年9月6日，消费金额：88888元"
    elif "李四" in query.lower():
        return f"客户记录：李四，普通客户，最近消费记录：2026年9月5日，消费金额：19元"
    else:
        return f"关于客户{query}无记录"

@tool
def send_email(customer:str) -> str:
    """
    发送感谢邮件
    :param query:
    :return:
    """
    return f"已向客户{customer}发送感谢邮件。"

class Customer_Analysis(BaseModel):
    """客户分析报告"""
    customer_name:str =Field(None,description="客户姓名")
    custmoer_level:Literal["潜在客户","普通客户","VIP客户","流失风险"] = Field("潜在客户",description="客户等级")
    record:str = Field(None,description="消费记录")
    spending_level:Literal["高","中","低"] = Field(None,description="消费水平")
    send_email:bool =Field(False,description="是否发送邮件")

agent = create_agent(
    model = model,
    system_prompt="""请分析客户的指定情况
    1.请先搜索客户数据库来查看客户信息
    2.如果是VIP客户则发送感谢邮件
    3.基于搜索结果生成结构化报告
    4.若用户提问与客户信息无关，或找不到客户信息，则返回空，不发送感谢邮件。
    
    """,
    tools = [search_database,send_email],
    response_format = ToolStrategy(Customer_Analysis)
)

result = agent.invoke({
    "messages" :[HumanMessage(content="请分析李四的客户信息")]
}

)
rprint(result)