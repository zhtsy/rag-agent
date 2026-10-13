from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Sequence
from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://127.0.0.1:11434",
    temperature=0.0
)
class MovieModel(BaseModel):
    """电影的详细信息"""
    title : str = Field(description="电影标题")
    year  : int = Field(description="发行年份")
    director  : str = Field(description="导演")
    rating : float = Field(description="电影评分，满分十分")

structured_llm = llm.with_structured_output(MovieModel)

result = structured_llm.invoke("请给出电影盗梦空间的信息")
print(result)