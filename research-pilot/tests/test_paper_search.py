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


# 非法年份
@pytest.mark.asyncio
@pytest.mark.parametrize("start_year, end_year, limit, error_msg",[(2026, 2023, 5, "不能晚于"), (2023, 2026, 101,"limit 必须在1到100之间")]) # normal, start_year > end_year, limit > 100 
async def test_search_papers_invalid_year_and_limit(start_year, end_year, limit, error_msg):
    """测试当起始年份大于终止年份时或limit>100时, 函数是否会按照预期抛出 ValueError """
    with pytest.raises(ValueError, match=error_msg):
        await search_papers(
            keyword="mmWave",
            start_year=start_year,
            end_year=end_year,
            limit=limit,
        )

# 没有搜索结果
@pytest.mark.asyncio
@respx.mock
async def test_search_papers_empty_results():
    """测试当搜索结果为空的时候(HTTP 200, results=[]), 函数应该返回空列表 []"""
    mock_response = {
            "meta": {"count": 0},
            "results": [
            ],
        }

    respx.get(
        "https://api.openalex.org/works"
        ).mock(
            return_value=httpx.Response(
                200,
                json=mock_response
            )
    )

    results = await search_papers(
        keyword="non_existent_random_keyword_xyz",
        start_year=2023,
        end_year=2024,
        limit=5,
    )

    # 3. 断言返回值是否为空列表且类型正确
    assert results == []
    assert isinstance(results, list)
    assert len(results) == 0

@pytest.mark.asyncio
@respx.mock
async def test_search_papers_timeout():
    """T05: 模拟网络请求超时，应抛出 httpx.TimeoutException"""
    # 模拟目标接口请求超时，抛出 TimeoutException
    # 替换为你实际请求的 URL 前缀或正则
    respx.get("https://api.openalex.org/works").mock(
        side_effect=httpx.TimeoutException("Connection timed out")
    )

    # 断言调用函数时抛出超时异常
    with pytest.raises(httpx.TimeoutException):
        await search_papers(
            keyword="mmWave",
            start_year=2023,
            end_year=2024,
            limit=5,
        )

@pytest.mark.asyncio
@respx.mock
async def test_search_papers_openalex_server_error():
    """模拟 OpenAlex服务故障, HTTP 500, 应抛出 httpx.HTTPStatusError"""
    respx.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(
                status_code=500,
                json={"error": "Internal Server Error"}
            )
        )

    with pytest.raises(httpx.HTTPStatusError):
        await search_papers(
            keyword="mmWave",
            start_year=2023,
            end_year=2024,
            limit=5,
        )

# 辅助函数 迷你 Responses API
import json 
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.paper_search_agent import (
    PaperSearchAgent,
)
from app.models.schemas import Paper

def make_response(output, output_text=""):
    return SimpleNamespace(
        output=output,
        output_text=output_text,
    )

def make_tool_call(
    name="search_papers",
    arguments=None,
    call_id="call_001",
):
    if arguments is None:
        arguments = {
            "keyword": "mmWave",
            "start_year": 2024,
            "end_year": 2026,
            "limit": 5,
        }

    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
        call_id=call_id,
        model_dump=lambda **kwargs: {
            "type": "function_call",
            "name": name,
            "arguments": json.dumps(arguments),
            "call_id": call_id,
        },
    )

# Test 1, no need tool
@pytest.mark.asyncio
async def test_agent_direct_answer():
    mock_client = MagicMock()

    mock_client.responses.create = AsyncMock(
        return_value=make_response(
            output=[],
            output_text="你好，我是 ResearchPilot",
        )
    )

    agent = PaperSearchAgent(
        client=mock_client,
        model="test-model",
        max_steps=5,
    )

    result = await agent.run("你好")
    assert result == "你好，我是 ResearchPilot"
    assert mock_client.responses.create.await_count == 1
    