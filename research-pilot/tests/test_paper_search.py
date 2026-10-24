import pytest
import httpx
import respx

from app.models.schemas import Paper
from app.tools.paper_search import search_papers


"""
自动化测试

1. 正常检索: 正常情况下进行论文检索
2. 非法关键词: 空字符串或者只有空格,应该抛出 ValueError
3. 非法年份与数量: 起始年份晚于结束年份，或者 limit=101,应该抛出 ValueError
4. 没有搜索结果: HTTP 200,results=[], 应该返回[] 
5. 网络请求超时: 模拟httpx.TimeoutException, 应该抛出超时异常
6. OpenAlex服务故障: 模拟HTTP 500, 应该抛出httpx.HTTPStatusError

"""

# 正常检测
@pytest.mark.asyncio
@respx.mock
async def test_search_papers_success():

    mock_response = {
        "meta": {"count": 1},
        "results": [
            {
                "id": "https://openalex.org/W123456789",
                "title": "Example Research Paper",
                "publication_year": 2025,
                "doi": None,
                "cited_by_count": 12,
            }
        ],
    }


    route = respx.get(
        "https://api.openalex.org/works"
    ).mock(
        return_value=httpx.Response(
            200,
            json=mock_response,
        )
    )

    papers = await search_papers(
        keyword="mmWave",
        start_year=2023,
        end_year=2026,
        limit=5,
    )

    assert route.called
    assert len(papers) == 1
    assert isinstance(papers[0], Paper)
    assert papers[0].title == "Example Research Paper"
    assert papers[0].year == 2025
    assert papers[0].doi is None
    assert papers[0].cited_by_count == 12

def test_paper_alias_mapping():
    raw_data = {
        "id": "https://openalex.org/W123456789",
        "title": "Test Paper",
        "publication_year": 2025,
        "doi": None,
        "cited_by_count": 10,
    }

    paper = Paper.model_validate(raw_data)

    assert paper.title == "Test Paper"
    assert paper.year == 2025
    assert paper.url == "https://openalex.org/W123456789"
    assert paper.cited_by_count == 10

    output = paper.model_dump()

    assert "year" in output
    assert "url" in output
    assert "publication_year" not in output
    assert "id" not in output


# 非法关键词检测
@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_keyword",["", "   ", None])
async def test_search_papers_invalid_keyword(invalid_keyword):
    """测试当关键词为空或者非法时，函数是否会按照预期抛出 ValueError"""
    with pytest.raises(ValueError, match="关键词不能为空"):
        await search_papers(
            keyword=invalid_keyword,
            start_year=2023,
            end_year=2026,
            limit=5,
        )