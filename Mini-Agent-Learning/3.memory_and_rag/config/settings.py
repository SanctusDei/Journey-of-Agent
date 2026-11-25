import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class Settings(BaseModel):
    """Model Settings"""

    api_key: str = Field(default_factory= lambda: os.environ.get("MODEL_API_KEY", ""))
    base_url: str = Field(default_factory= lambda: os.environ.get("MODEL_BASE_URL", ""))
    model_id: str = Field(default_factory= lambda: os.environ.get("MODEL_ID", ""))
    max_steps: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    is_stream: bool = Field(default=False)
    # max_history_turns: int = Field(default=10, ge=1)
    # prompt_path: str = Field(default="prompts/maid.md")

if __name__ == "__main__":
    
    settings = Settings()
    print(settings.__str__()) 
