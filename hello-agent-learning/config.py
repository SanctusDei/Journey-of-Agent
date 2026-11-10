"""配置管理"""
import os
from typing import Any
from pydantic import BaseModel, Field


class Config(BaseModel):
    """HelloAgent配置类"""

    # LLM Config
    default_model: str = "deepseek-flash"
    
    default_provider: str = "openai"

    temperature: float = Field(default=0.7, ge=0, le=2)

    max_tokens: int | None = Field(default=None, gt=0)

    # System Config
    debug: bool = False
    log_level: str = "INFO"

    # Other Config
    max_history_length: int = Field(default=100, gt=0)
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        
        return cls(
            
            debug= os.getenv("DEBUG", "false").lower() == "true",
            log_level= os.getenv("LOG_LEVEL", "INFO"),
            temperature= float(os.getenv("temperature", "0.7")),
            max_tokens= int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None

        )
        
    def to_dict(self) -> dict[str, Any]:
        """转化为字典"""

        return self.model_dump()

if __name__ == "__main__":
    """Test Config"""
    config = Config.from_env()

    print(config)
    print(config.to_dict())
