"""消息系统 基于（Response API）"""

from typing import Optional, Literal, Any
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field

# 自定义消息角色类型 
MessageRole = Literal["user", "assistant", "system", "developer"]

class Message(BaseModel):
    """Agent 内部消息模型"""

    content: str

    role: MessageRole
    
    timestamp: datetime = Field(
        default_factory= lambda: datetime.now(ZoneInfo("Asia/Shanghai"))    
    )

    metadata: dict[str, Any] = Field(default_factory=dict)

    # timestamp: datetime | None = None
    
    # metadata: Optional[dict[str, Any]] = None

    # def __init__(self, content: str, role: MessageRole, **kwargs):
    #     super().__init__(
    #             content = content,
    #             role = role,
    #             timestamp=kwargs.get('timestamp', datetime.now()),
    #             metadata=kwargs.get('metadata', {})
    #             )
    
    def to_dict(self) -> dict[str, Any]:
        """转化为字典格式(OpenAi API)"""

        return { 
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        
        return f"[{self.role}] {self.content}"

if __name__ == "__main__":
    """Test Message"""

    user_msg = Message(
        content= "什么是 Agent",
        role= "user",
        metadata={"source": "CLI"}
     )

    assistant_msg = Message(
        content="Agent 是能够利用模型与工具完成任务的系统。",
        role = "assistant"
    )

    print("=== 消息展示 ===")
    print(user_msg)
    print(assistant_msg)

    print("\n=== to_dict ===")
    print(user_msg.to_dict())

    print("\n=== matedata ===")
    print(user_msg.metadata)

    print("\n=== 时间戳 ===")
    print(user_msg.timestamp)




