from collections import deque

class WorkingMemory:

    def __init__(self, capacity: int = 3):
        self.messages = deque(maxlen=capacity)


    def add(self, message: str):
        self.messages.append(message)

    def get(self) -> list[str]:
        return list(self.messages)


if __name__ == "__main__":
    
    memory = WorkingMemory(capacity=3)

    for i in range(1, 6):
        memory.add(f"Message {i}")
        print(memory.get())
