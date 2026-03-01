from dataclasses import dataclass


@dataclass
class BaseContext:
    """
    基础上下文
    """

    system_prompt: str = ""
    tool: list = []

    def update(self, *kwargs):
        pass

    def get_context(self):
        pass

    def to_json(self):
        pass
