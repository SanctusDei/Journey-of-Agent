"""工具注册表 - Responses API"""

import json
from collections.abc import Callable
from typing import Any

from exceptions import HelloAgentsException
from .toolbase import Tool

class ToolRegistry:
    """HelloAgents 工具注册表

    支持:
    1. Tool 对象注册
    2. Python 函数注册
    3. Responses API Schema 生成
    4. 工具参数校验与执行
    5. Function Calling 结果构造
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}
        
    # ---- 工具注册 ----

    def register_tool(self, tool: Tool) -> None:
        """注册 Tool 对象"""
        
        if tool.name in self.list_tools():
            raise HelloAgentsException(
                f"工具 '{tool.name}' 已存在"
            )

        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[[str], str],
        parameters: dict[str, Any],
        ) -> None:
        """注册接受一个字符串参数的 Python 函数"""

        if name in self.list_tools():
            raise HelloAgentsException(
                f"工具 {name} 已经存在"
            )


        if not callable(func):
            raise TypeError("func 必须是可调用")

        if parameters.get("type") != "object":
            raise ValueError("parameters 必须是 object Schema")

        self._functions[name] = {
            "description": description,
            "func": func,
            "parameters":parameters,
        }

    def unregister(self, name: str) -> None:
        """注销工具。"""

        self._tools.pop(name, None)
        self._functions.pop(name, None)

    # ----- 工具查询 -----

    def get_tool(self, name: str) -> Tool | None:
        """获取 Tool 对象"""
        return self._tools.get(name)

    def get_function(
        self, name: str
        ) -> Callable[..., Any] | None:
        """获取注册的 Python 对象"""

        info = self._functions.get(name)
        return info["func"] if info else None


    def list_tools(self) -> list[str]:
        """返回所有工具名称"""

        return list(self._tools) + list(self._functions)

    def get_all_tools(self) -> list[Tool]:
        """返回所有 Tool 对象"""

        return list(self._tools.values())

    def get_tools_description(self) -> str:
        """生成便于人类阅读的工具描述"""
        descriptions = []

        for tool in self._tools.values():
            descriptions.append(
                 f"- {tool.name}: {tool.description}"
             )
 
        for name, info in self._functions.items():
            descriptions.append(
                f"- {name}: {info['description']}"
             )

        return "\n".join(descriptions) or "暂无可用工具"
    
    def get_tools_schema(self) -> list[dict[str, Any]]:
        """转换为 Response API 的 tools 格式"""
        
        schemas = []

        # Tool 对象: 根据 ToolParameter 生成 Schema
        for tool in self._tools.values():
            schemas.append(tool.to_response_schema())

        # 普通函数: 统一接受 Input 字符串
        for name, info in self._functions.items():
            schemas.append({
                "type": "function",
                "name": name,
                "description": info["description"],
                "strict": False,
                "parameters": info["parameters"] 
            })
        
        return schemas
    
    # ---- 参数验证 ----

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        """检查基本 JSON Schema 类型"""
        
        if expected == "string":
            return isinstance(value, str)

        if expected == "integer":
            return type(value) is int

        if expected == "number":
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        if expected == "boolean":
            return isinstance(value, bool)

        if expected == "array":
            return isinstance(value, list)

        if expected == "object":
            return isinstance(value, dict)

        raise ValueError(f"不支持的参数类型: {expected}")

    def _validate_tool_args(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        ) -> dict[str, Any]:
        """校验 Tool 对象参数并补充默认值"""

        params = {
            p.name: p for p in tool.get_parameters()
        }
        
        validated = dict(arguments)

        # 检查未知参数
        extra = set(validated) - set(params)
        if extra:
            raise ValueError(f"未知参数: {sorted(extra)}")

        # 检查必需参数和默认值
        for name, param in params.items():
            if name not in validated:
                if param.required:
                    raise ValueError(f"缺少必需参数: {name}")

                if param.default is not None:
                    validated[name] = param.default

        # 检查参数类型
        for name, value in validated.items():
            if not self._check_type(value, params[name].type):
                raise ValueError(
                    f"参数 '{name}' 类型错误，"
                    f"应为 {params[name].type}"
                )

        # 兼容 Tool 自定义校验逻辑
        if not tool.validate_parameters(validated):
            raise ValueError("工具参数校验失败")

        return validated

    # ---- 执行工具 ----

    def execute_tool(
        self,
        name:str,
        arguments: dict[str, Any]
    ) -> str:
        """执行指定工具, 返回字符串结果"""


        if not isinstance(arguments, dict):
            return "Error: 工具参数必须是字典"

        try:
            # 优先执行Tool对象
            if name in self._tools:
                tool = self._tools[name]

                validated = self._validate_tool_args(
                    tool, arguments
                )

                result = tool.execute(validated)

            # 执行普通 Python 函数
            elif name in self._functions:
                info = self._functions[name]
                schema = info["parameters"]

                properties = schema.get("properties", {})
                required = set(schema.get("required", []))

                # 检查缺失参数
                missing = required - set(arguments)

                if missing:
                    raise ValueError(
                        f"缺少参数: {sorted(missing)}"
                    )

                # 检查未知参数
                if schema.get("additionalProperties") is False:
                    extra = set(arguments) - set(properties)

                    if extra:
                        raise ValueError(
                            f"未知参数: {sorted(extra)}"
                        )

                # 检查基础类型
                for arg_name, value in arguments.items():
                    definition = properties.get(arg_name)

                    if definition is None:
                        continue

                    expected = definition.get("type")

                    if isinstance(expected, str):
                        if not self._check_type(value, expected):
                            raise ValueError(
                                f"参数 '{arg_name}' 类型错误"
                            )

                # 解包参数并执行函数
                func = info["func"]
                result = func(**arguments) 
            else:
                return f"错误: 未找到工具'{name}'"

                # 统一返回字符串
            if isinstance(result, str):
                return result

            return json.dumps(
               result,
               ensure_ascii=False,
            )
    
        except Exception as e:
            return f"错误: 工具'{name}'执行失败: {e}"

    def execute_call(self, call: Any) -> dict[str, str]:
        """执行模型返回的 function_call
        
        Args:
            call: Responses API 的function_call 对象

        Returns:
            可以直接回传给模型的 function_call_output
        """

        if call.type != "function_call":
            raise ValueError("输入不是 function_call 对象")

        try:
            arguments = json.loads(call.arguments)

            if not isinstance(arguments, dict):
                raise ValueError("arguments 必须是 JSON 对象")

            output = self.execute_tool(
                call.name, arguments
            )

        except (json.JSONDecodeError, ValueError) as e:
            output = f"错误: 工具参数解析失败: {e}"
        
        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": output,
        }

    # --- 管理 ————

    def clear(self) -> None:
        """清空注册表"""

        self._tools.clear()
        self._functions.clear()
    

        


            

            




if __name__ == "__main__":


    def add(a: int, b: int) -> int:
        return a + b


    def main():
        registry = ToolRegistry()

        registry.register_function(
            name="add",
            description="计算两个整数的和",
            func=add,
            parameters={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "第一个整数",
                    },
                    "b": {
                        "type": "integer",
                        "description": "第二个整数",
                    },
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        )

        print("工具名称:", registry.list_tools())

        print("执行结果:", registry.execute_tool(
            "add", {"a": 3, "b": 5}
        ))

        print("缺少参数:", registry.execute_tool(
            "add", {"a": 3}
        ))

        print("类型错误:", registry.execute_tool(
            "add", {"a": 3, "b": "hello"}
        ))
    main()


