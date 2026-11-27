import json
from pathlib import Path
from tempfile import TemporaryDirectory
from memory_item import MemoryItem
from dataclasses import asdict
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

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
        vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 2)
                )
        vectors = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, vectors)[0]
        print(scores) 
        return sorted(zip(items, scores), key= lambda x: x[1], reverse=True)[:limit]
        
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
        store.add(MemoryItem(content="你好,sdadadada"))
        store.add(MemoryItem(content="学习Agent"))

        # 测试重新加载
        new_store = MemoryStore(path)
        items = new_store.load()

        # 测试数量
        assert len(items) == 2

        # 测试反序列化
        assert isinstance(items[0], MemoryItem)

        # 测试内容

        print("All tests passed!")

        console.print(new_store.search(query="agent"))
        
        
