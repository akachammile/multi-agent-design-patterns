from langchain.tools import BaseTool
class ToolRegister:
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool):
        self.tools: BaseTool = tools
        self.tool_dict = {tool.name: tool for tool in tools}
    
    
    def add_tool(self, tool: BaseTool):
        if tool.name in self.tool_dict:
            return self

        self.tools += (tool,)
        self.tool_dict[tool.name] = tool
        return self

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_dict.get(name)
    
    
    
    
    