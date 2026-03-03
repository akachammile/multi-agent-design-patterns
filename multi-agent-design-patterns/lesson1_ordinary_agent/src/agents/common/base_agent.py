from .base_context import BaseContext


class BaseAgent:

    name: str = "base_agent"
    description: str = "底层代理"
    context: type[BaseContext] = BaseContext

    def __init__(self):
        pass

    def stream_messages(
        self,
        input,
    ):
        """
        流式输出后处理
        """
        pass
