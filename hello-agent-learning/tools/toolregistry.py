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
        ) -> None:
        """注册接受一个字符串参数的 Python 函数"""

        if name in self.list_tools():
            raise HelloAgentsException(
                f"工具 {name} 已经存在"
            )


        if not callable(func):
            raise TypeError("func 必须是可调用")

        self._functions[name] = {
            "desciption": description,
            "func": func,
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
        ) -> Callable[[str], str] | None:
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
                f"- {tool.name}: {info['desciption']}"
             )

        return "\n".join(descriptions) or "暂无可用工具"
    
    def get_tools_schma(self) -> list[dict[str, Any]]:
        """转换为 Response API 的 tools 格式"""

        schemas = []




if __name__ == "__main__":
    d2 = {"a": 3, "d": 2, "e": 2} 
    d = {"a": 1, "b": 2, "c": 3}
    print(list(d)+ list(d2))
