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
        pass
    
    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """获取工具参数定义"""
        pass

    def validate_parameters(
        self, paramters: dict[str, Any] 
    ) -> bool:
        """检查必需参数是否存在"""
        required = [
            p.name for p in self.get_parameters()
            if p.required
        ]

        return all(name in paramters for name in required)
    
    def to_dict(self) -> dict[str, Any]:
        """转化为内部工具描述"""
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                p.model_dump()
                for p in self.get_parameters()
            ]

        }
    
    def to_response_scchema(self) -> dict[str, Any]:
        """转化为 Response API Function Tool Schema"""

        params = {}

        properties = {}
        required = []

        for param in params:
            properties[param.name] = {
                "type": param.type, 
                "desciption": param.description,
            }

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "strict": False,
        }
    
    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()

if __name__ == "__main__":
    pass
