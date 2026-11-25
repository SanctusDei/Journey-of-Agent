from tools.base import ToolResult, Tool
from typing import Any

class ToolRegistry:
    """Tool Registry"""
    
    def __init__(self, tools: list[Tool] | None = None ) -> None:

        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)
    
    def register(self, tool: Tool) -> None:
        """register tools"""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already register")
        self._tools[tool.name] = tool

    def get(self,  name: str) -> Tool | None:
        """get tool""" 
        return self._tools.get(name)

    def get_schemas(self) -> list[dict[str, Any]]:
        """get openai schemas"""
        return [tool.to_openai_schema() for tool in self._tools.values()] 

    def tool_list(self) -> list[str]:
        """return registered tool's name"""
        return list(self._tools)

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """execute tool"""
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"tool '{name}' is not exist.")
        return await tool.execute(**kwargs)


if __name__ == "__main__":
    import asyncio 
    from tools import CalculatorTool
    from rich.console import Console
    
    console = Console()

    toolRegistry = ToolRegistry()
    
    calculator = CalculatorTool()
    
    toolRegistry.register(calculator)
    
    toolRegistry.show_tool_list()

    args = {"a": 1, "b": 2, "operator": "add"}

    console.print(toolRegistry.get_schemas())
    result = asyncio.run(toolRegistry.execute(calculator.name, **args))
    console.print(result)

    
    
    
