from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class MemoryItem:
    content: str
    memory_type: str = "semantic"
    importance: float = 0.5
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

if __name__ == "__main__":
    item = MemoryItem("用户使用 uv", importance=0.9)
    print(item)
