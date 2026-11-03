import os
import json
import inspect
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.tools.paper_search import search_papers
from app.tools.paper_rank import rank_papers
from app.tools.paper_detail import get_paper_detail
from app.models.schemas import TaskPlan
from app.agent.state import AgentState
from app.rag.retriever import retrieve_papers_semantic
from mcp import Client as MCPClient
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.markdown import Markdown
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

load_dotenv()
console = Console()

client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)

model = os.environ["OPENAI_MODEL"]

SEARCH_PAPERS_TOOL = {
    "type": "function",
    "name": "search_papers",
    "description": "通过 OpenAlex 检索学术论文。",
    "parameters": {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "论文检索关键词"
            },
            "start_year": {
                "type": "integer",
                "default": 2023
            },
            "end_year": {
                "type": "integer",
                "default": 2026
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 10
            }
        },
        "required": ["keyword"],
        "additionalProperties": False
    },
    "strict": False
}

RANK_PAPERS_TOOL = {
    "type": "function",
    "name": "rank_papers",
    "description": "对当前会话已保存的搜索结果按引用量排序，返回前 top_k 篇。",
    "parameters": {
        "type" : "object",
        "properties": {
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 3
            },    
        },
        "required": [],
        "additionalProperties": False
    },
    "strict": False,
}

# PAPER_DETAIL_TOOL = {
#     "type": "function",
#     "name": "get_paper_detail",
#     "description": "通过 OpeAlex Work ID 获取论文摘要和详细元数据。",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "paper_id": {
#                 "type":"string",
#                 "description": "OpenAlex Work ID，例如 W123456789。",
#             },
#         },
#         "required": ["paper_id"],
#         "additionalProperties": False,
#     },
#     "strict": False
# }

RETRIEVE_PAPERS_TOOL = {
    "type": "function",
    "name": "retrieve_saved_papers",
    "description": (
        "按语义检索当前搜索结果中已缓存的非空摘要。"
        "query 请使用英文。返回相关候选及覆盖数量；"
        "不会获取缺失详情，结果不代表全部符合条件的论文。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": False,
}


LOCAL_TOOLS = [
    SEARCH_PAPERS_TOOL,
    RANK_PAPERS_TOOL,
    RETRIEVE_PAPERS_TOOL
]

# 使用Langgraph

class SearchGraphState(TypedDict):
    keyword: str
    paper_count: int
    result: dict



class PaperSearchAgent:
    def __init__(self, client: AsyncOpenAI, model: str, max_steps: int = 5):
        self.client = client
        self.model = model
        self.max_steps = max_steps
        # 新增状态
        self.state = AgentState()
        # 工具注册表
        self.tool_registry = {
            "search_papers": self.search_and_save,
            "rank_papers": self.rank_saved_papers,
            "get_paper_detail": self.detail_and_save,
            "retrieve_saved_papers": self.retrieve_saved_papers,
        }
        # 嵌入层
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        self.tools = LOCAL_TOOLS.copy()
        
    async def run(self, query: str, target_papers:int = 0) -> str:
        
        if not 0 <= target_papers <= 5:
            raise ValueError("target_papers 必须在0到5之间")

        self.state.target_papers = target_papers
        self.state.plan = []

        if target_papers > 0:
            # 生成计划
            self.state.plan = await self.create_plan(query)
            self.state.selected_ids = []

            console.print("[cyan]本次任务计划[/cyan]")
            for i, item in enumerate(self.state.plan, start=1):
                console.print(f"{i}. {item}") 

        # 将State 摘要加入系统消息
        state_summary = json.dumps(
            self.state.snapshot(),
            ensure_ascii=False,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是 ResearchPilot 科研助手。"
                    "只能依据工具返回的真实论文信息回答。"
                    "需要检索时调用 search_papers；"
                    "需要排序时调用 rank_papers。"
                    f"\n当前会话状态：{state_summary}"
                    "需要从当前候选论文的缓存摘要中查找相关内容时，"
                    "调用 retrieve_saved_papers，检索词使用英文。"
                    "需要获取缺失摘要时，调用 get_paper_detail。"
                    "回答时说明可检索摘要数和候选论文数，"
                    "区分检索相关性与论文内容中的明确证据。"
                    "区分候选论文数、参与检索的摘要数和实际返回数；"
                    "不要把 Top-K 返回数量称为检索覆盖数量。"
                    "按当前状态中的plan推进任务，并依据工具结果检查进度。"
                    "已查询但没有摘要的论文不要反复查询；证据不足时明确说明。"
                    "search_papers 会替换当前候选集合，不会合并历次搜索结果。"
                    "没有缓存摘要不代表论文没有摘要；必须区分尚未查询和已查询但摘要为空。"
                    "数量、标题和链接以状态及工具结果为准，不要编造合并去重过程。"
                    "历史对话用于理解指代和整理已有回答。"
                    "新增或核实论文事实时，以最新状态和工具结果为准。"
                ),
            },
            *self.state.chat_history,
            {"role": "user", "content": query}
            ]

        # Agent Loop
        for step in range(self.max_steps):
            console.print(f"\n[cyan]第 {step + 1} 轮模型调用[/cyan]")

            # 1. 询问模型
            responses = await self.client.responses.create(
                model=self.model, input=messages, tools=self.tools
            )

            # 2. 保存模型输出
            messages.extend(
                item.model_dump(exclude_none=True) for item in responses.output
            )

            # 3. 提取工具调用
            tool_calls = [
                item for item in responses.output if item.type == "function_call"
            ]

            # 4. 如果模型不再调用工具，说明任务完成
            if not tool_calls:
                answer =  responses.output_text or ""

                if self.state.target_papers > 0:
                    progress = self.check_progress()
                    if progress["status"] != "ready":
                        answer =  (
                            f"摘要证据尚未齐备：{progress}\n\n"
                            f"{answer}"
                        )

                self.state.remember_turn(query, answer)
                return answer
            
            # 5. 执行工具
            for tool_call in tool_calls:
                tool_output = await self._execute_tool(tool_call)

                # 6. 将工具结果加入消息历史
                messages.append(tool_output)

            if self.state.target_papers > 0:
                progress = self.check_progress()
                console.print(f"摘要准备进度: {progress}")

                messages.append({
                    "role": "system",
                    "content": "Python核对的最新摘要准备进度:"
                    + json.dumps(progress, ensure_ascii=False),
                })
 
        answer = f"达到最大执行轮数 {self.max_steps},Agent 停止运行"
        self.state.remember_turn(query, answer)
        return answer

    async def _execute_tool(self, tool_call) -> dict:
        """执行工具"""
        try:

            if tool_call.name not in self.tool_registry:
                raise ValueError(f"未注册的工具: {tool_call.name}")

            # json -> python dict
            args = json.loads(tool_call.arguments)

            if not isinstance(args, dict):
                raise ValueError("工具参数必须是 JSON 对象")

            console.print(f"[dim]模型选择工具：{tool_call.name}，参数：{args}[/dim]")

            # 获取Tool方法
            tool_func = self.tool_registry.get(tool_call.name)
            
            # 兼容同步和异步工具
            result = tool_func(**args)

            if inspect.isawaitable(result):
                result = await result

            # Pydantic 模型与普通dict 统一为 JSON
            # Pydantic & dict -> Json 兼容的 Python dict

            if isinstance(result, list):
                result = [
                    item.model_dump(mode="json")
                    if hasattr(item, "model_dump") 
                    else item
                    for item in result
                ]
            # 将 result list[dict] -> json str 
            output = json.dumps(result, ensure_ascii=False)

        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            output = json.dumps(
                {"error": str(exc)},
                ensure_ascii=False
            )

        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": output,
        }

    async def search_and_save(
        self,
        keyword: str,
        start_year: int = 2023,
        end_year:int = 2026,
        limit: int = 10,
    ) -> dict:
        papers = await search_papers(
            keyword=keyword,
            start_year=start_year,
            end_year=end_year,
            limit=limit,
        )

        # 保存完整论文数据到 State
        self.state.papers = {
            paper.url: paper for paper in papers
        }

        self.state.selected_ids = []
        self.state.last_keyword = keyword

        # 只返回简短结果给模型
        return {
            "update_mode": "replace",
            "count": len(papers),
            "preview": [
                {
                    "url": p.url,
                    "title": p.title,
                    "year": p.year,
                    "cited_by_count": p.cited_by_count,
                }
                for p in papers[:10]
            ],
        }
    
    async def detail_and_save(self, paper_id:str) -> dict:
        # 将短 ID 和完整 URL 统一成相同的缓存键。
        work_id = paper_id.strip().rstrip("/").split("/")[-1]
        paper_url = f"https://openalex.org/{work_id}"

        if paper_url in self.state.paper_details:
            console.print(f"[dim]读取详情缓存：{work_id}[/dim]")
            return self.state.paper_details[paper_url]

        console.print(f"[dim]通过 MCP 获取详情：{work_id}[/dim]")

        async with MCPClient("http://localhost:8000/mcp") as tool_client:
            result = await tool_client.call_tool(
                "get_paper_detail",
                {"paper_id": paper_id}
            )

        if result.is_error:
            raise ValueError(f"MCP 工具执行失败：{result.content}")

        detail = result.structured_content

        if not isinstance(detail, dict):
            raise ValueError("MCP 论文详情工具未返回字典数据")

        self.state.paper_details[paper_url] = detail

        return detail

    async def create_plan(self, query:str) ->list[str]:
        """生成计划"""
        response = await self.client.responses.create(
            model=self.model,
            tools=[],
            input=[
                {
                    "role":"system",
                    "content":(
                        "将科研任务拆成最多6个简短、按顺序执行的待办步骤。"
                        "可用能力：搜索论文、引用量排序、获取摘要、摘要检索、比较总结。"
                        "只生成计划，不编造论文ID和研究结果。"
                        '仅输出JSON：{"steps": ["步骤1", "步骤2"]}。'
                        "不要使用Markdown代码块。"
                    ),
                },
                {"role": "user", "content": query}
            ],
        )
        parsed = TaskPlan.model_validate_json(response.output_text)
        return parsed.steps

    async def load_mcp_tools(self) -> None:
        async with MCPClient("http://localhost:8000/mcp") as client:
            result = await client.list_tools()

        for tool in result.tools:
            if tool.name != "get_paper_detail":
                continue

            model_tool = {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
                "strict": False,
            }

            self.tools = [
                *LOCAL_TOOLS,
                model_tool
            ]

            console.print(f"已从 MCP 加载工具: {tool.name}")
            console.print(
                "当前模型可用工具:",
                [ item["name"] for item in self.tools ],
            )
            return
        raise ValueError("MCP 服务未提供 get_paper_detail 工具")
    
    def check_progress(self) -> dict:
        # 已经选中的论文ID
        ids = self.state.selected_ids
        # 需要比较的论文数量
        target = self.state.target_papers
        # 已经有详细查询记录的论文数量(不一定获取到了摘要)
        queried = 0
        # 已经获取
        abstracts = 0

        for paper_id in ids:
            if paper_id in self.state.paper_details:
                queried += 1

            abstract = self.state.paper_details.get(
                paper_id, {}
            ).get("abstract")

            # 检查摘要是否有效
            if isinstance(abstract, str) and abstract.strip():
                abstracts += 1

        status = "pending"
        if target > 0 and len(ids) == target and queried == target:
            status = "ready" if abstracts == target else "blocked"

        return {
            "target": target,
            "selected": len(ids),
            "queried": queried,
            "abstracts": abstracts,
            "status": status,
        }

    def rank_saved_papers(self, top_k: int = 3) -> dict:
        if not self.state.papers:
            raise ValueError("没有搜索结果，请先搜索论文")

        if not 1 <= top_k <= 100:
            raise ValueError("top_k 必须在 1 到 100 之间")
        
        ranked = sorted(
            self.state.papers.values(),
            key=lambda p: p.cited_by_count,
            reverse=True
        )[:top_k]

        self.state.selected_ids = [
            paper.url for paper in ranked
        ]

        return {
            "selected_papers": [
                {
                    "url": p.url,
                    "title": p.title,
                    "year": p.year,
                    "cited_by_count":p.cited_by_count,
                }
                for p in ranked
            ]
        }

    def retrieve_saved_papers(self, query: str, top_k: int = 3) -> dict:
        if not query.strip():
            raise ValueError("query 不能为空")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在1到10之间")

        searchable = {
            paper_id: detail
            for paper_id, detail in self.state.paper_details.items()
            if paper_id in self.state.papers
            and isinstance(detail.get("abstract"), str)
            and detail["abstract"].strip()
        }

        results = retrieve_papers_semantic(
            paper_details=searchable,
            query=query,
            model=self.embedding_model,
            top_k=top_k,
        )
        # 将摘要检索结果与已有论文元数据关联
        for item in results:
            paper = self.state.papers[item["paper_id"]]

            item.update({
                "title": paper.title,
                "year": paper.year,
                "url": paper.url,
                "cited_by_count": paper.cited_by_count,
            })


        return {
            "query": query,
            "requested_top_k": top_k,
            "candidate_count": len(self.state.papers),
            "searchable_count": len(searchable),
            "returned_count": len(results),
            "results": results,
        }

    def build_search_graph(self):
        """构建搜索流程图"""
        async def search_node(state: SearchGraphState):
            console.print("[cyan]执行节点：search[/cyan]")

            data = await self.search_and_save(
                keyword=state["keyword"],
                limit=5,
            )

            return { "paper_count": data["count"] }

        def choose_next(state: SearchGraphState):
            return "rank" if state["paper_count"] > 0 else "empty"

        def rank_node(state: SearchGraphState):
            console.print("[cyan]执行节点：rank[/cyan]")

            data = self.rank_saved_papers(top_k=2)
            return {"result": data}

        def empty_node(state: SearchGraphState):
            console.print("[cyan]执行节点：empty[/cyan]")

            return {
                "result": {"message": "本次搜索没有找到候选论文"}
            }

        builder = StateGraph(SearchGraphState)
        builder.add_node("search", search_node)
        builder.add_node("rank", rank_node)
        builder.add_node("empty", empty_node)

        builder.add_edge(START, "search")
        builder.add_conditional_edges(
            "search",
            choose_next,
            {"rank": "rank", "empty": "empty"},
        )
        builder.add_edge("rank", END)
        builder.add_edge("empty", END)

        return builder.compile()
    

if __name__ == "__main__":
    import asyncio
    from pathlib import Path

    async def main():
        
        state_path = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "state.json"
        )

        # 整个交互过程只创建一次Agent
        agent = PaperSearchAgent(
            client=client,
            model=model,
            max_steps=8,
        )

        # 加载MCP_tool

        await agent.load_mcp_tools()

        # 读取存储的state
        agent.state = AgentState.load(state_path)

        console.print(    
            f"已恢复 {len(agent.state.papers)} 篇候选论文,"
            f"{len(agent.state.paper_details)} 条详情缓存"
        )

        try:
            # 没有候选论文时，执行首次科研任务
            if not agent.state.papers:
                answer = await agent.run(
                    query=(
                        "搜索2023—2026年 radar gesture recognition 论文，最多5篇。"
                        "在本次候选中按引用量选出前2篇，获取它们的摘要。"
                        "根据摘要比较识别方法，附标题和链接。"
                        "没有摘要时明确说明，不要推测论文方法。"
                    ),
                    target_papers=2,
                )
                agent.state.save(state_path)
                console.print(Markdown(answer))

            while True:
                query = input("\n继续提问，输入 exit 结束：").strip()

                if query.lower() == "exit":
                    break

                if not query:
                    continue
                console.print(f"本次收到的问题：{query!r}")
                answer = await agent.run(query)
                agent.state.save(state_path)
                console.print(Markdown(answer))

        finally:
            agent.state.save(state_path)
    asyncio.run(main())



"""
1. 什么是 Function Calling? 模型会直接执行 Python 函数吗？

Function Calling 是模型根据用户需求，生成结构化的工具调用请求。
模型只负责决定调用哪个工具及传入什么参数，真正的函数执行由
Agent Harness(Python 程序)完成。

2. Responses API 中 response.output 返回什么?

response.output 是一个列表，其中可能包含不同类型的 Pydantic 对象。

例如：
- ResponseOutputMessage:模型生成的文本消息。
- ResponseFunctionToolCall:模型生成的工具调用请求。

因此，需要通过 item.type == "function_call" 筛选工具调用。

3. 为什么要使用 json.loads(tool_call.arguments)?

因为 tool_call.arguments 虽然内容是 JSON，
但其 Python 类型实际上是 str。

json.loads() 将 JSON 字符串转换成 Python 字典，
使其可以通过 **args 的方式传给工具函数

4. 为什么要 TOOL_REGISTRY？

TOOL_REGISTRY 是工具名称到 Python 函数的映射表。

例如：
{"search_papers": search_papers}

模型返回的是工具名称字符串，而不是 Python 函数对象。
Harness 需要通过注册表找到对应函数并执行。

5. 为什么异步工具需要 await?

async def 定义的函数在被调用时返回 coroutine 对象，
不会立即执行完整的异步逻辑。

必须通过 await 或其他 asyncio 调度机制运行协程，
才能获得工具执行结果。

6. 为什么需要 message.extend(response.output)?

因为模型生成的工具调用也属于对话历史。

下一轮模型请求需要知道：
- 用户提出了什么问题。
- 模型发起了什么工具调用。
- 工具返回了什么结果。

对于无状态的 Responses API，需要手动维护完整历史。

7. model_dump(exclude_none=True) 有什么用?

model_dump() 将 Pydantic 对象转换为 Python 字典。

exclude_none=True 表示转换时省略值为 None 的字段。

它有利于消息序列化和调试，但不保证转换后的字典
一定符合所有第三方 API 的输入格式。

8. 为什么需要 function_call_output 和 call_id ?

function_call_output 用于将 Python 工具的执行结果返回模型。

call_id 是工具调用的关联标识。

对应的 function_call 和 function_call_output
必须使用相同的 call_id，服务端才能正确匹配。

9. 什么是 Agent Loop ?

Agent Loop 是模型决策和工具执行的循环过程:
模型决策 → 执行工具 → 更新消息历史 → 再次请求模型。

当模型不再请求工具时，Harness 可以结束循环，
返回最终回答。

10. 为什么 Agent Loop 需要 max_steps?

因为模型可能持续请求调用工具。

max_steps 限制模型调用轮数，避免无限循环，
并帮助控制 Token 消耗、运行时间和 API 成本。

11. 为什么使用 AsyncOpenAI 而不是 OpenAI?

AsyncOpenAI 支持通过 await 发起异步模型请求，
可以与 async def search_papers() 使用统一的异步架构。

但异步并不等于自动并行，顺序 await 仍然是顺序执行。

12. 为什么将 run() 和 _execute_tool() 分开？
这是职责分离（Separation of Concerns）。

run() 负责：
- Agent Loop
- 模型决策
- 消息历史管理
- 任务终止判断

_execute_tool() 负责：
- 解析工具参数
- 查找工具函数
- 执行工具
- 序列化工具结果

这样可以降低代码耦合，并方便独立测试。

13. Agent = Model + Harness 应该如何理解?

Model 负责决策:
- 是否调用工具?
- 调用哪个工具?
- 使用什么参数？
- 是否继续执行?

Harness 负责执行:

- 管理模型请求和消息历史
- 分发并执行工具
- 处理工具返回结果
- 控制循环、错误和终止条件

模型提供决策能力，Harness 提供执行环境和控制逻辑。

14. 当前 ResearchPilot 属于什么阶段?

已经实现最小的多轮 Tool Calling Agent 架构：

- Responses API
- Function Calling
- Tool Registry
- Agent Loop
- max_steps
- run() / _execute_tool() 职责拆分

下一阶段需要补充：
- 工具异常处理
- Pydantic 参数校验
- Mock 单元测试
- 多工具注册与扩展
- Agent 执行日志

"""


"""
Day 4

Q1: State 和 Context 的区别

State:Agent 在程序中维护的状态，可能包含完整论文数据、工具结果、任务进度等。
Context: 一次模型调用中实际提供给 LLM 的信息，包括系统指令、用户消息、工具调用历史和必要的状态摘要等

Q2: 为什么 rank_papers() 不再需要 papers 参数?

由于共享状态(Shared State), rank_papers 可以访问同一个 State 参数

Q3: default_factory=list 的作用

每次创建一个新的 AgentState 实例, 调用一次 list(), 生成一个新的空列表

Q4: Agent 如何回答'刚才选中的第一篇是什么?'

全流程: 上一轮:rank_papers()选出论文 --> State 保存 Selected_ids --> snapshot() 提取论文信息
--> 将摘要放进 Context --> LLM 读取Context 并回答

"""

"""
Day 7

Q1. State 与持久化记忆是什么关系?
State 是运行时使用的数据；持久化将其中需要保留的部分保存下来，供下次启动恢复。

Q2. 为什么不能把整个Agent保存为JSON?
Agent 包含模型客户端、函数和 Embedding 模型等对象。今天只保存论文元数据和摘要等可转换为 JSON 的数据。

Q3. 加载JSON后, 为什么还要调用 Paper.model_validata()?
JSON 加载得到普通字典，需要恢复成现有代码使用的 Paper 对象。

Q4. 恢复文件后，模型是否自动知道全部摘要？
不会。摘要恢复到了 Python State 中，仍要通过状态摘要或工具结果提供给模型。

Q5. 今天的持久化能否从上次中断的工具步骤继续执行?
不能。当前保存的是论文数据，执行计划、循环位置和完整对话历史没有恢复。

"""

"""
Day 8

Q1. 论文缓存和联通记录分别解决什么问题?
论文缓存保存研究材料；聊天记录帮助理解“刚才”“上一条”等连续追问。

Q2. 保存了 chat_history, 模型就能直接读取吗?
答：不能，还需要将它加入本次请求的 messages .

Q3. 为什么记录最终回答,而不是只记录用户问题
答：“把刚才的回答改成表格”需要知道助手之前具体回答了什么。

Q4. 为什么只保留最近三轮?
先用简单方式限制历史长度。问答很长时，仍然可能占用较多 Token，之后再学习按长度裁剪和摘要压缩。

Q5. 历史回答能否直接当成新的论文证据?
不能。历史回答可能存在错误；新增或核实事实时，应依据最新状态和工具返回的论文信息。

"""

"""
Day 8
Q1 MCP 和 Function Calling 分别做什么?
Function Calling 让模型提出工具调用；MCP 让应用通过标准接口发现和调用工具服务。

Q2 @mcp.tool 和 call_tool() 有什么区别?
前者查询有哪些工具，后者实际执行指定工具。

Q3 引入MCP后,摘要缓存保存在哪里?
仍保存在 Agent 的 state.paper_details 中。MCP 服务负责获取并返回详情。

Q4 为什么先检查 is_error, 再读取结果?
调用返回了结果对象，不代表工具执行成功。失败结果不能当作正常论文详情保存。

Q5 list_tools() 会执行论文查询吗?
不会。它返回工具定义，实际执行使用 call_tool()

Q6 MCP 的 input_schema 应放到模型工具定义的哪个字段？
parameters

Q7 为什么已经发现工具，任然需要注册执行方法?
工具定义告诉模型如何提出调用；执行方法负责真正调用服务、处理结果和更新 State。

Q8 为什么再首次run() 前加载工具?
模型调用时，self.tools 必须已经包含论文详情工具的描述。

Q9 服务端新增一个工具，Agent就能自动执行它吗?
当前版本还不能。本节只加载已接入的论文详情工具；新增工具还需要接好对应的执行路径。。

"""

"""
Day 9

Q1 LangGraph 的三个基本组成部分是什么?
State 保存数据，Node 执行工作，Edge 决定后续路径。

Q2 节点需要返回整个状态吗?
不需要。可以只返回需要更新的字段，其他字段保留。

Q3 普通边与条件边有什么区别?
普通边指定固定的下一节点；条件边根据当前状态选择下一节点。

Q4 为什么本节项目使用 ainvoke()
搜索节点包含异步请求，需要通过异步方式运行整张图。

Q5 用了LangGraph,流程就会自动具备模型自主决策能力吗?
不会。自主决策需要接入模型节点，并根据模型输出设置工具执行和循环分支。

"""