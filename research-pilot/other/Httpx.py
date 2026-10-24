import httpx
import asyncio
# response = httpx.get('https://www.example.org/')
# print(response)
# print(response.status_code)
# print(response.headers['content-type'])
# print(response.text)

# r = httpx.get('https://api.github.com/events')
# print()
# print(r.json())

# data = {'key1': 'value1', 'key2': 'value2'}
# r = httpx.post("https://httpbin.org/post", data=data)
# # print(r.text)
# async def main():
#     async with httpx.AsyncClient() as client:
#         r = await client.get('https://www.example.com/')

OPENALEX_URL="https://api.openalex.org/works"

# params: dict[str, str | int] = {
#         "search": keyword.strip(),
#         "filter": f"publication_year:{start_year}-{end_year}",
#         "per_page": limit,
#     }

import asyncio
import httpx

OPENALEX_URL = "https://api.openalex.org/works"

params = {
    "search": "machine learning",
    "filter": "publication_year:2023-2025",
    "per_page": 1,
    # 4. 可选过滤条件（例如年份、开源状态等）
    # "select": "id,title,publication_year,doi,cited_by_count",
}

async def main():
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            OPENALEX_URL,
            params=params,
        )

        response.raise_for_status()
        data = response.json()
        # print(response)
        # print()
        print(data.get("results", []))

if __name__ == "__main__":
    asyncio.run(main())