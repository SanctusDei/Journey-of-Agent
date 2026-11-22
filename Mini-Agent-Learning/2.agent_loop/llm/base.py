from abc import ABC, abstractmethod
from typing import Any

class LLMClientBase(ABC):
    """Abstract base class for llm client"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
    ):
        self.api_key = api_key
        self.api_base = api_base
