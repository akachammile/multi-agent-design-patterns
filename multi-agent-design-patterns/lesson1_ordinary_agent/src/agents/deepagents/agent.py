from .context import DeepAgentContext
from src.agents.common import BaseAgent, load_model
from src.configs import config as sys_config
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph


class DeepAgent(BaseAgent):

    name = "deep_agent"
    description = "深度代理"
    context = DeepAgentContext

    def __init__(self):
        pass

    def get_agent(self) -> CompiledStateGraph:
        agent = create_agent(
            model=load_model(sys_config.default_model),
            system_prompt=self.context.system_prompt,
        )
        return agent
