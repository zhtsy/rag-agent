from langchain.agents import create_agent
from langchain.tools import tool
import os

from langchain_core.messages import HumanMessage

# 你的 search_knowledge_base 已经是一个 @tool 了，直接导入
from rag_agent import search_knowledge_base

@tool
def list_skills() -> str:
    """列出所有可用的 Skill。"""
    return "\n".join(os.listdir("./skills"))

@tool
def load_skill(skill_name: str) -> str:
    """加载指定 Skill 的完整内容。
    重要：skill_name 必须是 list_skills 返回的技能文件夹名称（英文，如 knowledge-base-search），
    不要翻译成中文，不要使用描述文字。"""
    path = f"./skills/{skill_name}/SKILL.md"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return f"Skill {skill_name} 不存在。请使用 list_skills 返回的英文名称。"

agent = create_agent(
    model="ollama:qwen2.5:7b",
    tools=[list_skills, load_skill, search_knowledge_base],
    system_prompt="""你是一个助手。
    当用户询问人物信息或喜好时，你必须先使用 list_skills 查看可用技能，
    然后使用 load_skill 加载对应技能，再按照技能指令调用 search_knowledge_base 工具。"""
)

response = agent.invoke({"messages": [HumanMessage(content="我是小智，我喜欢什么")]})

# 打印完整消息，看是否有 ToolMessage
for msg in response["messages"]:
    print(type(msg).__name__, getattr(msg, 'content', '')[:200])
    print("tool_calls:", getattr(msg, 'tool_calls', None))
    print("-" * 40)