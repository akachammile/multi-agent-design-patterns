import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parents[4]  # 向上4层到达项目根目录
sys.path.insert(0, str(project_root))
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from settings import settings, LLM
from langgraph.graph import START, MessagesState, END

from typing import TypedDict


llm = LLM()


class Agent_A_State(MessagesState):
    pass


class Agent_A_Output(BaseModel):
    output: str = Field(default="", description="输出内容")


def agent_a():
    create_agent(model=llm)
