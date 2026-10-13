from typing import Optional, Literal

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "qwen2.5:7b",
    base_url = "http://127.0.0.1:11434",
    temperature = 0.0
)



class CustommerInfo(BaseModel):
    name: str = Field(description="客户姓名")
    phone: str = Field(description="客户手机号")
    email: Optional[str] = Field(description="客户邮箱")
    issue: str = Field(description="情况描述")
    urgency: Literal["低","中","高"] = Field(description="紧急程度")

structured_llm = llm.with_structured_output(CustommerInfo)


conversation = """
客服：先生请问你有什么事？
客户：我是小平，我的手机号是18888888888，我的货怎么还没到，我瘾犯了，我很急。
客服：好的我马上去查一下。

"""
result = structured_llm.invoke(f"从一下客服对话信息中获取客户信息：\n{conversation}")
print(result)


