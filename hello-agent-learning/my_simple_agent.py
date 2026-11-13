from typing import Optional, Iterator
from agents import SimpleAgent
from hello_agents import ToolRegistry
from core import Config, Message, HelloAgentsLLM

class MySimpleAgent(SimpleAgent):
    """
    重写的简单对话Agent
    展示如何基于框架基类构建自定义Agent
    """
    
    def __init__(
        self,
        name:str,
        llm:HelloAgentsLLM,
        system_prompt: str | None = None,
        config: Config() = None,
        tool_registry: ToolRegistry | None = None,
        enable_tool_calling:bool = True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"{name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")
        
    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        重写的运行方法 - 实现简单的对话逻辑, 支持可选工具调用
        """
        print(f"{self.name} 正在处理: {input_text}")

        # 构建消息列表
        messages = []

        # 添加系统信息(可能包含工具信息)
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})


        # 添加历史消息
        for msg in self._history:
            messages.append(msg.to_dict())

        # 添加当前用户信息
        messages.append({"role": "user", "content": {input_text}})

        # 如果没有使用工具调用，使用简单的对话逻辑
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"{self.name} 响应完成")
            return response

        # 支持多轮工具调用的逻辑
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词, 包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt

        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":

            return base_prompt

        tools_section = "\n\n ## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题：\n"
        tools_section += tools_description + "\n"

        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如:`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"
 
        return base_prompt + tools_section 
    
    def _run_with_tools(self, messagess:list, input_text: str, max_tool_iterations: int, **kwargs) -> str:
        """支持工具调用的运行逻辑"""
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            
            # 调用LLM
            response = self.llm.invoke(message)

            # 检查是否有工具调用
            tool_calls = self._parse_tool_calls(response)
