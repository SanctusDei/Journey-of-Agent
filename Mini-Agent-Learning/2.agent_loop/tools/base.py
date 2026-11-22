"""Base Tool Class"""
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel

class ToolResult(BaseModel):
    """Tool execute result"""
    # success
    success: bool
    content: str = ""
    error: str | None = None

class Tool(ABC):
    """Base Class for all tools"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass 

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Tool parameters schema (JSON Schema format)"""
    @abstractmethod
    async def execute(self, *args, **kwargs) -> ToolResult:
        """Execute the tool with arbitary arguments"""
        pass

    def to_openai_schema(self)  -> dict[str, Any]:
        """Convert tool to OpenAI tool schema."""
        return {
            "type": "function",
            "name": self.name,
            "strict": True,
            "description": self.description,
            "parameters":self.parameters 
        }

if __name__ == "__main__":
    """Test Tool"""
    import asyncio
    from rich.console import Console
    console = Console()
    class AddTool(Tool):

        @property
        def name(self) -> str:

            return "add"

        @property
        def description(self) -> str:
            return "return the result of a + b"       

        @property
        def parameters(self) -> dict[str, Any]:

            return {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "int",
                        "description": "the first number"
                    },
                    "b": {
                        "type": "int",
                        "description": "the second number"
                    }
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            }
        async def execute(self, a: int, b: int) -> ToolResult:

            return ToolResult(success=True, content=str(a+b))

    console.print(ToolResult.model_json_schema())
    async def main():
        add_tool = AddTool()
        console.print(add_tool.name)
        console.print(add_tool.description)
        console.print(add_tool.parameters)
        console.print(add_tool.to_openai_schema())

        # 测试实际执行工具
        result = await add_tool.execute(a=3, b=5)
        console.print(result)

    asyncio.run(main()) 
