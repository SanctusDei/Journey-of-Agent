"""工具基类"""

from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel

class ToolParameter(BaseModel):
    """工具参数定义"""

    name:str
    type:Literal[
        "string", "integer", "number",
        "boolean", "array", "object"
    ]
    desciption: str 
    required: bool = True
    default: Any = None

class Tool(ABC):
    """工具基类"""

    def __init__(self, name:str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, parameters: dict[str, Any]) -> str:
        """执行工具"""
    
    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """获取工具参数定义"""

    def validate_parameters(
        self, paramters: dict[str, Any] 
    ) -> bool:
        """检查必需参数是否存在"""
        required = [
            p.name for p in self.get_parameters()
            if p.required
        ]

        return all(name in paramters for name in required)
    
if __name__ == "__main__":
    pass
