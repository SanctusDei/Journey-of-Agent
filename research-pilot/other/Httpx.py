import httpx
import asyncio
from rich.console import Console
from app.models.schemas import Paper
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

console = Console()

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
    "per_page": 5,
    # 4. 可选过滤条件（例如年份、开源状态等）
    "select" : "id,title,publication_year,doi,cited_by_count"
}

async def main():
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            OPENALEX_URL,
            params=params,
        )

        response.raise_for_status()
        # console.print(type(response), response) # <class 'httpx.Response'> <Response [200 OK]>
        data = response.json()
        results = data.get("results", [])
        papers: list[Paper] = [Paper.model_validate(item) for item in results]
        # console.print(papers)
        console.print(type(papers[0]))

        # console.print(data, type(data))

        # # print(response)
        # # print()


if __name__ == "__main__":


    """
                {
            'meta': {
                'notice': 'No API key, so this request shares one free daily budget ($0.10) with everyone on your network\'s IP address. Get 
        a free key with its own $1/day at https://openalex.org/settings/api and add api_key=YOUR_KEY to your requests. An AI agent can get 
        one for its user in one call: POST https://api.openalex.org/api-keys with {"email": "<your user\'s email>"}; we email them a link, 
        and the key works once they click it. Docs for agents: https://help.openalex.org/llms.txt',
                'count': 1922527,
                'db_response_time_ms': 214,
                'page': 1,
                'per_page': 1,
                'groups_count': None,
                'x_query': {
                    'oql': 'get works from 2023–2025 with (machine learning) in the full text',
                    'x_new_oql': 'get works from 2023–2025 with (machine learning) in the full text',
                    'oqo': {
                        'get_rows': 'works',
                        'filter_rows': [
                            {'column_id': 'fulltext.search', 'value': 'machine learning', 'operator': 'has'},
                            {'column_id': 'publication_year', 'value': 2023, 'operator': '>='},
                            {'column_id': 'publication_year', 'value': 2025, 'operator': '<='}
                        ]
                    },
                    'url': '/works?filter=fulltext.search:machine learning,publication_year:2023-2025&per_page=1',
                    'order': 'best matches first',
                    'order_default': True
                },
                'cost_usd': 0.001
            },
            'results': [
                {
                    'id': 'https://openalex.org/W4388698868',
                    'title': 'Machine learning for microbiologists',
                    'publication_year': 2023,
                    'doi': 'https://doi.org/10.1038/s41579-023-00984-1',
                    'cited_by_count': 262
                }
            ],
            'group_by': []
        }
        <class 'dict'>
                """
    asyncio.run(main())
     