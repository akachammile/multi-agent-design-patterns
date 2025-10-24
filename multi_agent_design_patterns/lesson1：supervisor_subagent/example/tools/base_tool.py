from langchain_core.tools import BaseTool


class CustomTool(BaseTool):
    name = "Custom Tool"
    description = "A custom tool that does something"

    def _run(self, query: str) -> str:
        return "Custom tool response"

    async def _arun(self, query: str) -> str:
        return "Custom tool response"