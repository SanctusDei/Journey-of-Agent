"""

调用方式


tool_call.name = "rank_papers"

tool_call.arguments = '''
{
    "papers": [
        {
            "title": "论文 A",
            "cited_by_count": 10,
            "url": "https://openalex.org/W1"
        },
        {
            "title": "论文 B",
            "cited_by_count": 100,
            "url": "https://openalex.org/W2"
        }
    ],
    "top_k": 1
}
json str
'''



"""
from rich.console import Console

console = Console()

def rank_papers(
    papers: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """按照论文引用量从高到低排序"""

    if not 1 <= top_k <= 100:
        raise ValueError("top_k 必须在 1 到 100之间")

    return sorted(
        papers,
        key=lambda p: p.get("cited_by_count") or 0,
        reverse=True,
    )[:top_k]


def test_rank_papers():

    """测试rank_papers函数"""
    papers = [
        {"title": "论文 A", "cited_by_count": 20},
        {"title": "论文 B", "cited_by_count": 150},
        {"title": "论文 C", "cited_by_count": 60},
        {"title": "论文 D", "cited_by_count": 5},
        {"title": "论文 E", "cited_by_count": 100},
    ]

    result = rank_papers(papers, top_k=3)
    console.print(result)

    assert [p["title"] for p in result] == ["论文 B", "论文 E", "论文 C"]
    assert [p["cited_by_count"] for p in result] == [150, 100, 60]    


if __name__ == "__main__":
    test_rank_papers()
