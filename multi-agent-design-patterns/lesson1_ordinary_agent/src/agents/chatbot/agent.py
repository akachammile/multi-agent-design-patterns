from .context import Context
from langchain.agents import create_agent
from src.agents.common import BaseContext, BaseAgent


class ChatbotAgent(BaseAgent):
    """Chatbot agent"""

    name: str = "chatbot_agent"
    description: str = "Chatbot agent"
    context: type[BaseContext] = Context
