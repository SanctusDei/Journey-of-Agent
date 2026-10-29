import pytest

from app.tools.paper_rank import rank_papers


def test_rank_papers():
    """乱序输入时，默认返回引用量最高的三篇论文。"""
    papers = [
        {"title": "论文 A", "cited_by_count": 20},
        {"title": "论文 B", "cited_by_count": 150},
        {"title": "论文 C", "cited_by_count": 60},
        {"title": "论文 D", "cited_by_count": 5},
        {"title": "论文 E", "cited_by_count": 100},
    ]

    result = rank_papers(papers)

    assert [paper["title"] for paper in result] == ["论文 B", "论文 E", "论文 C"]
    assert [paper["cited_by_count"] for paper in result] == [150, 100, 60]
    # sorted() 返回新列表，原列表的顺序应保持不变。
    assert [paper["title"] for paper in papers] == [
        "论文 A", "论文 B", "论文 C", "论文 D", "论文 E"
    ]


def test_rank_papers_missing_citations():
    """引用量缺失或为 None 时按零排序；top_k 可超过论文数量。"""
    papers = [
        {"title": "引用量缺失"},
        {"title": "引用量为空", "cited_by_count": None},
        {"title": "有引用", "cited_by_count": 10},
        {"title": "零引用", "cited_by_count": 0},
    ]

    result = rank_papers(papers, top_k=10)

    assert [paper["title"] for paper in result] == [
        "有引用", "引用量缺失", "引用量为空", "零引用"
    ]


def test_rank_papers_empty():
    assert rank_papers([]) == []


@pytest.mark.parametrize("top_k", [0, 101])
def test_rank_papers_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k 必须在 1 到 100 之间"):
        rank_papers([], top_k=top_k)
