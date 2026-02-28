from dataclasses import dataclass


@dataclass
class BaseContext:
    """
    基础上下文
    """

    system_prompt: str = ""
    history: list = []

    def __init__(self):
        pass

    def get_context(self):
        pass
