from .context import Context
from src.agents.common import load_chat_model
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from src.agents.common import BaseContext, BaseAgent
from langgraph.graph.state import CompiledStateGraph


class ChatbotAgent(BaseAgent):
    """Chatbot agent"""

    name: str = "chatbot_agent"
    description: str = "Chatbot agent"
    context: type[BaseContext] = Context

    def get_graph(self, **kwargs) -> CompiledStateGraph:
        """
        获取并编译对话图实例。
        必须确保在编译时设置 checkpointer，否则将无法获取历史记录。
        例如: graph = workflow.compile(checkpointer=sqlite_checkpointer)
        """
        graph = create_agent(
            model=load_chat_model(
                "gemini/gemini-3-flash-preview"
            ),  # 默认模型，会被 middleware 覆盖
            checkpointer=await self._get_checkpointer(),
        )

        self.graph = graph
        return graph
