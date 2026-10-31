import os
import httpx
from dotenv import load_dotenv

from app.models.schemas import Paper

"""

要求:
1.根据关键词搜索论文
2.按照发表日期范围进行筛选
3.限制返回数量
4.提取标题、年份、DOI、引用数等
5.输入参数校验
6.网络超时与HTTP错误处理
7.无结果时返回空列表
8.支持API Key 配置
9.支持排序和分页

"""

load_dotenv()

OPENALEX_URL = "https://api.openalex.org/works"

async def search_papers(
    keyword: str,           # search
    start_year: int = 2023, # filter
    end_year: int = 2026,   # filter
    limit: int = 10,        # limit
) -> list[Paper]:

    # TODO 1: 校验输入参数
    if not keyword or not keyword.strip():
        raise ValueError("关键词不能为空！")

    if start_year > end_year:
        raise ValueError(f"起始年份({start_year}) 不能晚于 ({end_year}).")

    if not 1 <= limit <= 100:
        raise ValueError("limit 必须在1到100之间。")

    # TODO 2: 构建 OpenAlex 查询参数
    params = {
        "search": keyword,
        "filter": f"publication_year:{start_year}-{end_year}",
        "per_page": limit,
        "select" : "id,title,publication_year,doi,cited_by_count"
    }
    openalex_key = os.getenv("OPENALEX_API_KEY")
    # TODO 3: 从环境变量读取 API Key（可选）
    if openalex_key:
        params["api_key"] = openalex_key
    # TODO 4: 使用 httpx.AsyncClient 发起 GET 请求

    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(
            OPENALEX_URL,
            params=params,
        )

    # TODO 5: 检查 HTTP 响应状态
        response.raise_for_status()
    # TODO 6: 解析 response.json()
        data = response.json()
    # TODO 7: 将 results 中每条记录转换成 Paper
        results = data.get("results", [])
        papers: list[Paper] = [Paper.model_validate(item) for item in results]

    # TODO 8: 返回 list[Paper]
        return papers


"""
Day-1 知识点:

1.为什么search_papers() 使用 async def?

因为 HTTP 请求涉及网络 I/O。asyncio 可以在请求等待期间调度其他任务；但单独使用 async 并不会自动让多个请求并行。

2.为什么需要response.raise_for_status()?

HTTPX 默认不会因为 404、500 等状态码自动抛出状态异常。调用该方法后，失败状态会触发 HTTPStatusError。

3.为什么Agent 不应该使用原始的JSON?

通过 Paper 统一字段、处理缺失值和校验类型，可以减少外部 API 格式对后续 Agent 的影响。

4.为什么需要Mock测试?

它可以模拟成功、空结果、超时和服务错误，使测试不依赖网络、外部 API 状态及每日调用额度。

"""