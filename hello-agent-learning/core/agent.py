"""Agent基类"""
from abc import ABC, abstractmethod
from typing import Optional, Any
from .message import Message
from .config import Config
from hello_agents import HelloAgentsLLM

class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name:str,
        llm:HelloAgentsLLM | None = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config(),
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs:Any) -> str:
        """运行Agent, 由子类实现"""
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""

        self._history.append(message)

    def clear_history(self) -> None:
        """清空历史记录"""

        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""

        return self._history.copy()

    def __str__(self) -> str:
        
        return f"Agent(name={self.name})"


if __name__ == "__main__":
    """Test Agent"""
    from rich.console import Console

    class SimpleAgent(Agent):

        def run(self, text: str) -> str:
            self.add_message(text)
            return f"{self.name}: 收到消息 {text}"
    agent = SimpleAgent("TestAgent")
    print(agent.run("你好"))
    print(agent._history)
    print(agent)
