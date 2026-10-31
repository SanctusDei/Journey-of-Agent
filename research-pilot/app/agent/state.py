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
            "selected_papers": selected_papers

        }