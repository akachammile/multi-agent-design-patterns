import sys
from pathlib import Path

# 把项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 根据你的文件位置调整
sys.path.append(str(PROJECT_ROOT))


from llm import LLM
from pydantic import Field
from langchain.agents import create_agent

class BaseAgent:
    """base agent for creat subagent
    """
    def __init__(self, tools, system_prompt, middleware=None):
        self.llm = LLM()
        self.tools = tools
        self.system_prompt = system_prompt
        self.middleware = middleware or []

    def create(self):
        return create_agent(
            model=self.llm.client,
            tools=self.tools,
            system_prompt=self.system_prompt,
            middleware=self.middleware
        )