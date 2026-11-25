from abc import ABC, abstractmethod
from collections.abc import Callable
from core.llmbase import LLMClientBase, OpenAIResponseClient
from tools.registry import ToolRegistry
from tools.CalculatorTool import CalculatorTool
from typing import Any
import json

class BaseAgent(ABC):
    """Base class for all Agent"""

    @abstractmethod
    async def run(
        self,
        user_input:str,
        on_chunk: Callable[[str], None] | None = None
    ) -> str:
        """start loop once"""

    @abstractmethod
    def clear(self) -> None:
        """clear histroy"""

class MaidAgent(BaseAgent):
    """ base on ReAct / Tool-Calling """

    def __init__(
        self,
        llm: LLMClientBase,
        tools: ToolRegistry,
        instructions: str,
        max_steps: int = 5,
        is_stream: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.instructions = instructions
        self.max_steps = max_steps
        self.is_stream = is_stream
        self.history: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.history.clear()

    async def run(
        self,
        user_input: str,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:

        self.history.append({"role": "user", "content": user_input})

        for _ in range(self.max_steps):
            response = await self.llm.chat(
                messages=self.history,
                instructions=self.instructions,
                tools=self.tools.get_schemas(),
                stream=self.is_stream,
                on_chunk=on_chunk
            )

            if response.text: 
                self.history.append({"role": "assistant", "content": response.text})

            if not response.tool_calls:
                return response.text

            for tc in response.tool_calls:
                self.history.append({
                    "type": "function_call",
                    "call_id": tc.call_id,
                    "name": tc.name,
                    "arguments": tc.arguments
                })

                args = json.loads(tc.arguments)
                result = await self.tools.execute(tc.name, **args)
                output = result.content if result.success else f"Error: {result.error}"

                self.history.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": str(output)
                })

        return f"抱歉主人，思考超过{self.max_steps}步任然为未完成任务。"

if __name__ == "__main__":


    import asyncio
    from config.settings import Settings
    from core.llmbase import OpenAIResponseClient
    from tools.registry import ToolRegistry
    from tools.CalculatorTool import CalculatorTool
    from prompts.loader import load_prompt
    from rich.markdown import Markdown
    from rich.console import Console

    try:
        import readline  # noqa: F401 - Enables multibyte line editing for input().
    except ImportError:
        pass

    # rich print
    console = Console()
    
    # load config
    settings = Settings()
    
    # tools
    calculator = CalculatorTool()
    
    # tool register
    toolRegistry = ToolRegistry()
    toolRegistry.register(calculator) 
    # tools_schemas
    
    # prompt
    prompt = load_prompt("prompts/maid.md")


    # create llm
    llm = OpenAIResponseClient(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_id,
    )
   
    # create agent
    maidagent = MaidAgent(
        llm = llm,
        tools = toolRegistry,
        instructions=prompt,
    )
    async def main():
        while True:
            query = input("主人,请输入内容(输入exit退出):")

            try:
                query.encode("utf-8")
            except UnicodeEncodeError:
                console.print("输入中有损坏的字符，请重新输入。")
                continue

            if not query.strip():
                continue
            if query.strip() == "exit":
                break

            output_text = await maidagent.run(
                    user_input= query
            )

            console.print(Markdown(output_text))

    asyncio.run(main())




    
