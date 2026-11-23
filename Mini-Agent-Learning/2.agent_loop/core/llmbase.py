from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable
from dataclasses import dataclass, field
from openai import AsyncOpenAI

@dataclass
class ToolCall:
    """common function call """
    call_id: str
    name: str
    arguments: str

@dataclass
class LLMResponse:
    """common model response"""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

class LLMClientBase(ABC):
    """Abstract base class for llm client"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        instructions: str,
        tools: list[dict[str, Any]],
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> LLMResponse:
       """ start to chat and return LLMResponse """ 


class OpenAIResponseClient(LLMClientBase):
    """base on OpenAi Response API"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        ) -> None:
        
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model
        self.temperature = temperature

    async def chat(
        self,
        messages: list[dict[str, Any]],
        instructions: str,
        tools: list[dict[str, Any]],
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None
        ) -> LLMResponse:
        
        params = {
            "model": self.model,
            "instructions": instructions,
            "input": messages,
            "tools": tools,
            "temperature": self.temperature
        }

        if not stream:
            response = await self.client.responses.create(**params)
            return self._parse_response(response)

        response_stream = await self.client.responses.create(**params, stream=True)
        final_response = None

        async for event in response_stream:
            if event.type == "response.output_text.delta" and on_chunk:
                on_chunk(event.delta)
            elif event.type == "response.completed":
                final_response = event.response

        return self._parse_response(final_response) if final_response else LLMResponse()

    def _parse_response(self, response: Any) -> LLMResponse:
        tool_calls = [
            ToolCall(
                call_id=item.call_id,
                name=item.name,
                arguments=item.arguments
            )
            for item in (response.output or [])
            if item.type == "function_call"
        ]

        return LLMResponse(text=response.output_text or "", tool_calls= tool_calls)


if __name__ == "__main__":
   
    import json
    from config.settings import Settings
    from tools.CalculatorTool import CalculatorTool
    from tools.registry import ToolRegistry
    from prompts.loader import load_prompt
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    settings = Settings()

    # Tools
    calculator = CalculatorTool()


    client = OpenAIResponseClient(
        settings.api_key,
        settings.base_url,
        settings.model_id, 
    )

    prompt = load_prompt("prompts/maid.md") 
    toolRegistry = ToolRegistry()
    toolRegistry.register(calculator)

    tools_list = toolRegistry.tool_list() 

    if tools_list:
        print("Avaliabe Tools:")
        print("".join(f"· {name}\n" for name in tools_list ), end="")
    

    async def main():

        history = []
        step = 0

        while True:

            query = input("请输入你想询问的内容(exit):")
            if query.strip() == "exit":
                break

            if not query.strip():
                continue

            history.append({"role": "user", "content": query})
            
            while step < settings.max_steps:

                response = await client.chat(
                    messages = history,
                    instructions= prompt,
                    tools= toolRegistry.get_schemas(),
                    stream=False
                )

                if response.text: 

                    console.print(Markdown(response.text)) 
                    history.append({"role": "assistant", "content": response.text})
                if not response.tool_calls:
                    break

                for toolcall in response.tool_calls:
                    history.append({
                        "type": "function_call",
                        "name": toolcall.name,
                        "call_id": toolcall.call_id,
                        "arguments": toolcall.arguments
                    })
                for toolcall in response.tool_calls:

                    args = json.loads(toolcall.arguments)
                    fun_name = toolcall.name

                    # Tool Result 
                    result = await toolRegistry.execute(fun_name, **args)
                    output_str = result.content if result.success else f"error: {result.error}"
                    print(f"[cyan]tool call: {toolcall.name}({toolcall.arguments}) -> {output_str}[/cyan]")
                    
                    history.append({
                        "type": "function_call_output",
                        "call_id": toolcall.call_id,
                        "output": str(output_str)
                    })

                step += 1
    
    import asyncio
    asyncio.run(main())
            
                
                
                


        
        

        

        


        
        
        

        




