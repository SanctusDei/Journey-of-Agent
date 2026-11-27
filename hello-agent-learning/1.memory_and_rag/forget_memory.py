

def forget(memories: list[dict], threshold: float):
    return [m for m in memories if m["importance"] >= threshold]


if __name__ == "__main__":

    memories = [
        {"content": "你好", "importance": 0.1},
        {"content": "正在学习 Agent", "importance": 0.7},
        {"content": "使用 uv", "importance": 0.9}
    ]

    print(forget(memories, threshold=0.5))
