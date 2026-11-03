from mcp.server import MCPServer
from app.tools.paper_detail import (
    get_paper_detail as fetch_paper_detail
)

mcp = MCPServer("ResearchPilot Tools")

@mcp.tool()
async def get_paper_detail(paper_id: str) -> dict[str, str]:
    """通过 OpenAlex Work ID 或完整 URL 获取论文摘要"""
    return await fetch_paper_detail(paper_id)