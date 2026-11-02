import json
from pathlib import Path
from dataclasses import dataclass, field
from app.models.schemas import Paper

@dataclass
class AgentState:
    # 保存完整的论文对象
    papers: dict[str, Paper] = field(default_factory=dict)

    # 保存选中的论文 URL
    selected_ids: list[str] = field(default_factory=list)

    paper_details: dict[str, dict] = field(default_factory=dict)

    last_keyword: str = ""
    # 计划
    plan : list[str] = field(default_factory=list)
    target_papers: int = 0
    
    # 聊天历史
    chat_history: list[dict[str, str]] = field(default_factory=list)

    def snapshot(self) -> dict:
        
        selected_papers = []

        for paper_id in self.selected_ids:
            paper = self.papers.get(paper_id)
            if paper is None:
                continue

            item = {
                "url": paper_id,
                "title": paper.title,
                "year": paper.year,
                "doi": paper.doi,
                "cited_by_count": paper.cited_by_count,
            }

            
            abstract = self.paper_details.get(paper_id, {}).get("abstract")
            item["has_abstract"] = isinstance(abstract, str) and bool(abstract.strip())

            selected_papers.append(item)

    
        return {
            
            "paper_count": len(self.papers),
            "last_keyword": self.last_keyword,
            "selected_papers": selected_papers,
            "plan": list(self.plan),
            "target_papers": self.target_papers,
        }

    def save(self, path: str | Path) -> None:
        data = {
            "papers": [
                {
                    "id": paper.url,
                    "title": paper.title,
                    "publication_year": paper.year,
                    "doi": paper.doi,
                    "cited_by_count": paper.cited_by_count,
                }
                for paper in self.papers.values()
            ],
            "selected_ids": self.selected_ids,
            "paper_details": self.paper_details,
            "last_keyword": self.last_keyword,
            "chat_history":self.chat_history,
        }

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AgentState":
        
        file_path = Path(path)

        if not file_path.exists():
            return cls()

        data = json.loads(
            file_path.read_text(encoding="utf-8")
        )

        papers = [
            Paper.model_validate(item)
            for item in data.get("papers", [])
        ]

        return cls(
            papers={paper.url: paper for paper in papers},
            selected_ids=data.get("selected_ids", []),
            paper_details=data.get("paper_details", {}),
            last_keyword=data.get("last_keyword", ""),
            chat_history=data.get("chat_history", [])[-6:],
        )

    def remember_turn(self, query: str, answer: str) -> None:
        self.chat_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ])

        # 每轮两条消息，只保留最近三轮
        self.chat_history = self.chat_history[-6:]