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
    description: str 
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
        self, parameters: dict[str, Any] 
    ) -> bool:
        """检查必需参数是否存在"""
        required = [
            p.name for p in self.get_parameters()
            if p.required
        ]

        return all(name in parameters for name in required)
    
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
    
    def to_response_schema(self) -> dict[str, Any]:
        """转化为 Response API Function Tool Schema"""


        properties = {}
        required = []

        for param in self.get_parameters():
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
    import json
    from rich.console import Console
    
    class CalcuatorTool(Tool):
        """计算器工具:支持加、减、乘、除"""

        def __init__(self):
            super().__init__(
                name="calculator",
                description="执行基本数学运算",
            )
        
        def get_parameters(self) -> list[ToolParameter]:
            return [
                ToolParameter(
                    name="a",
                    type="number",
                    description="第一个数字"
                ),
                ToolParameter(
                    name="b",
                    type="number",
                    description="第二个数字"
                ),
                ToolParameter(
                    name="operation",
                    type="string",
                    description="运算类型: add/subtract/multiply/divide"
                ),
                
            ]
        
        def execute(self, parameters: dict[str, Any]) -> str:
            a = parameters["a"]
            b = parameters["b"]
            operation = parameters["operation"]

            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return "Error: 除数不能为0"
                result = a / b
            else:
                return f"Error : 不支持的运算 {operation}"
            
            return str(result)
    console = Console()

    # 1. 实例化工具
    tool = CalcuatorTool()

    # 2. 生成工具 Schema
    schema = tool.to_response_schema()

    console.print(json.dumps(schema, indent=2, ensure_ascii=False))

    # 3. 模拟 LLM 返回的function_call
    function_call = {
            "name": "calculator",
            "arguments": json.dumps({
                "a": 12.5,
                "b": 8,
                "operation": "multiply"
            }),
    }

    # 4. 解析工具调用
    tool_name = function_call["name"]
    arguments = json.loads(function_call["arguments"])

    # 5. 查找工具并执行
    if tool_name == tool.name:
        if tool.validate_parameters(arguments):
            result = tool.execute(arguments) 
            print("工具执行结果:", result)
        else:
            print("缺少参数")
    

