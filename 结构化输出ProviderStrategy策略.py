from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("ollama:qwen2.5:7b")


class InformationModel(BaseModel):
    name: str = Field(description="用户姓名")
    email: str = Field(description="用户电子邮箱")
    phone: str = Field(description="用户手机号码")


agent = create_agent(
    model=model,
    response_format=ProviderStrategy(InformationModel)
)

response = agent.invoke({
    "messages": [HumanMessage(
        content="从以下信息中提取用户信息。我叫小明，我的邮箱是dawdawdjawjadawjdaj@qq.com，我的电话号码是1888888888888.")]

}

)

print(response)
