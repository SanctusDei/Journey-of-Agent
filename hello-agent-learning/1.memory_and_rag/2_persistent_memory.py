import json
from pathlib import Path
from tempfile import TemporaryDirectory
from memory_item import MemoryItem
from dataclasses import asdict
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2", device="cpu"
        )


    def load(self) -> list[MemoryItem]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [MemoryItem(**item) for item in data]

    def add(self, item: MemoryItem) -> None:
        items = self.load()
        items.append(item)
        self._save(items)

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:

        items = self.load()
        if not items:
            return []
        documents = [item.content for item in items]
        embeddings = self.model.encode(documents)
        query = self.model.encode_query(query)
        scores = self.model.similarity(embeddings, query).squeeze(1).tolist()
        console.print(scores)
        ranked = sorted(zip(items, scores), key= lambda x : x[1], reverse= True)[:limit]
        return [item for item, _ in ranked]      

    def _save(self, items: list[MemoryItem]) -> None:
        data = [asdict(item) for item in items]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8"
        )
        


if __name__ == "__main__":
    console = Console()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        store = MemoryStore(path)

        # 测试空记忆
        assert store.load() == []

        # 测试添加记忆
        # 测试添加记忆
        store.add(MemoryItem(content="用户喜欢使用 Python 开发人工智能应用。"))
        store.add(MemoryItem(content="用户习惯使用 Vim 编辑代码。"))
        store.add(MemoryItem(content="用户正在学习大语言模型智能体的记忆机制。"))
        store.add(MemoryItem(content="用户喜欢在周末打篮球和跑步。"))
        store.add(MemoryItem(content="用户正在研究如何通过向量数据库实现语义检索。"))        


        # 测试重新加载
        new_store = MemoryStore(path)
        items = new_store.load()

        # 测试数量

        # 测试反序列化
        assert isinstance(items[0], MemoryItem)

        # 测试内容

        print("All tests passed!")

        console.print(new_store.search(query="用户正在学习哪些AI技术"))
        
        
