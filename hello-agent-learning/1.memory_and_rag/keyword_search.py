
def search(query: str, memories: list[str]) -> list[str]:
    if not query.strip():
        return []
    return [m for m in memories if query.lower() in m.lower()]

if __name__ == "__main__":

    memories = [
        "使用 uv 管理 Python 依赖",
        "使用 pytest 进行 Python 测试",
        "正在学习二叉树"
    ]

    print(search("pytest", memories=memories))

