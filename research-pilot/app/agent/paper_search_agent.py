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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from openai.types.responses import ResponseFunctionToolCall
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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

class AgentGraphState(TypedDict):
    """智能体执行流程的状态结构"""

    messages: list
    tool_calls: list[dict]
    answer: str
    steps: int
    query: str
    business_state: dict

class PaperSearchAgent:

    def __init__(self, client: AsyncOpenAI, model: str, max_steps: int = 5, checkpointer=None):
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

        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.agent_graph = self.build_agent_graph()
        
    async def run(
        self,
        query: str,
        target_papers:int = 0,
        thread_id: str = "main",   
        ) -> str:
        """ 启动新任务，准备任务计划和模型消息，并通过状态图执行。

        Args:
            query: 用户的查询或任务描述。
            target_papers: 目标论文数量, 取值为 0-5: 为 0 时不生成任务计划。
            thread_id: 执行任务的线程ID, 用于关联检查点。
        
        Returns:
            智能体生成的最终回答。
        """

        # 校验目标论文数量的取值范围
        if not 0 <= target_papers <= 5:
            raise ValueError("target_papers 必须在0到5之间")

        # 构建线程配置，并获取该线程最近保存的状态快照
        config = self._graph_config(thread_id)
        previous = await self.agent_graph.aget_state(config)
    
        # 如果仍有待执行节点，则要求先恢复已有任务
        # 避免在同一线程中启动新任务而覆盖未完成的执行状态
        if previous.next:
            raise ValueError("存在未完成任务，请先调用 resume()")

        # 设置本次任务的目标论文数量，并清空上一轮任务计划
        self.state.target_papers = target_papers
        self.state.plan = []

        # 指定目标论文数量时，根据用户查询生成计划
        if target_papers > 0:
            # 生成计划
            self.state.plan = await self.create_plan(query)

            # 清空此前选中的论文 ID, 准备本次任务的论文选择
            self.state.selected_ids = []

            # 按顺序展示本次任务计划
            console.print("[cyan]本次任务计划[/cyan]")
            for i, item in enumerate(self.state.plan, start=1):
                console.print(f"{i}. {item}") 

        # 将当前业务状态的摘要转为 JSON, 作为模型的上下文
        # ensure_ascii=False 保留中文等非 ASCII 字符，便于阅读
        state_summary = json.dumps(
            self.state.snapshot(),
            ensure_ascii=False,
        )

        # 构建模型输入: 系统指令、历史对话、本次用户调查
        messages = [
            {
                "role": "system",
                # 规定工具使用方式、证据要求、数量统计口径和缓存规则
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
                    "论文详情缓存按 OpenAlex URL 复用；"
                    "更换搜索关键词不代表同一论文的缓存摘要失效。"
                    "需要报告缓存覆盖数量时，以工具返回的数据为准。"
                ),
            },
            # 展开历史对话，帮助模型理解上下文和指代
            *self.state.chat_history,
            # 将本次调查追加为最新的用户消息
            {"role": "user", "content": query}
            ]

        # Agent Loop
        # for step in range(self.max_steps):
        #     console.print(f"\n[cyan]第 {step + 1} 轮模型调用[/cyan]")

        #     # 1. 询问模型
        #     responses = await self.client.responses.create(
        #         model=self.model, input=messages, tools=self.tools
        #     )

        #     # 2. 保存模型输出
        #     messages.extend(
        #         item.model_dump(exclude_none=True) for item in responses.output
        #     )

        #     # 3. 提取工具调用
        #     tool_calls = [
        #         item for item in responses.output if item.type == "function_call"
        #     ]

        #     # 4. 如果模型不再调用工具，说明任务完成
        #     if not tool_calls:
        #         answer =  responses.output_text or ""

        #         if self.state.target_papers > 0:
        #             progress = self.check_progress()
        #             if progress["status"] != "ready":
        #                 answer =  (
        #                     f"摘要证据尚未齐备：{progress}\n\n"
        #                     f"{answer}"
        #                 )

        #         self.state.remember_turn(query, answer)
        #         return answer
            
        #     # 5. 执行工具
        #     for tool_call in tool_calls:
        #         tool_output = await self._execute_tool(tool_call)

        #         # 6. 将工具结果加入消息历史
        #         messages.append(tool_output)

        #     if self.state.target_papers > 0:
        #         progress = self.check_progress()
        #         console.print(f"摘要准备进度: {progress}")

        #         messages.append({
        #             "role": "system",
        #             "content": "Python核对的最新摘要准备进度:"
        #             + json.dumps(progress, ensure_ascii=False),
        #         })
 
        # answer = f"达到最大执行轮数 {self.max_steps},Agent 停止运行"
        # self.state.remember_turn(query, answer)
        # return answer

        # 初始化本次执行的图状态, 交由状态图驱动方法运行
        # 模型调用、工具执行、中断交互和结果收尾由状态图以及驱动方法负责
        return await self._drive_graph(
            {
                "query": query,        # 本次用户查询
                "messages": messages,  # 系统指令、历史对话和当前查询
                "tool_calls": [],      # 待执行的工具调用，初始为空
                "answer": "",          # 最终回答，初始为空
                "steps": 0,            # 已执行的模型调用轮数，从 0 开始
                # 将当前业务对象序列化，供图节点读取和更新.
                "business_state": self.state.to_dict(),  
            },
            config,
        )
        
    async def _execute_tool(self, tool_call) -> dict:
        """执行模型请求的工具，并将结果或指定异常封装为工具输出消息。"""
        try:
            # 检查工具是否已经注册，避免调用不存在的工具
            if tool_call.name not in self.tool_registry:
                raise ValueError(f"未注册的工具: {tool_call.name}")

            # json -> python dict
            # 将模型传入的JSON 参数字符串解析为 Python 对象
            args = json.loads(tool_call.arguments)

            # 参数必须为字典, 才能通过 **args 作为关键词参数传入工具
            if not isinstance(args, dict):
                raise ValueError("工具参数必须是 JSON 对象")

            # 展示模型选择的工具以及调用参数
            console.print(f"[dim]模型选择工具：{tool_call.name}，参数：{args}[/dim]")

            # 从注册表中获取工具函数
            tool_func = self.tool_registry.get(tool_call.name)
            
            # 调用工具: 同步函数直接返回结果，异步函数通常返回协程对象
            result = tool_func(**args)

            # 如果返回值可以等待，则等待执行完成，获取实际结果
            if inspect.isawaitable(result):
                result = await result

            # Pydantic 模型与普通dict 统一为 JSON
            # Pydantic & dict -> Json 兼容的 Python dict

            # 如果结果为列表, 将其中具有 model_dump 方法的原生
            # 按 Pydantic 模型的方式转为 JSON 兼容数据， 其余元素保持原样
            if isinstance(result, list):
                result = [
                    item.model_dump(mode="json")
                    if hasattr(item, "model_dump") 
                    else item
                    for item in result
                ]
            # 将 result list[dict] -> json str
            
            # 将工具结果序列化为 JSON 字符串, 保留中文字符  
            output = json.dumps(result, ensure_ascii=False)

        except (json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            # 将上述类型的参数解析、工具执行或序列化异常转为错误信息
            # 错误信息会作为工具结果返回给模型, 供其后续处理
            output = json.dumps(
                {"error": str(exc)},
                ensure_ascii=False
            )

        # 构造工具输出消息, 通过 call_id 关联模型发起的那次工具调用
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

    def build_agent_graph(self):        
        """构建并编译包含主题澄清、模型推理、工具执行和结果收尾的智能体状态图"""

        #校验最大模型调用轮数，确保至少执行一轮
        if self.max_steps < 1:
            raise ValueError("max_steps 必须至少为 1")

        def clarify_node(state: AgentGraphState):
            """检查查询是否缺少研究主题，必要时暂停流程并向用户询问"""

            # 需要进一步澄清研究主题的常见模糊查询
            vague_queries = {
                "帮我找论文",
                "请帮我找论文",
                "帮我搜索论文",
                "请帮我搜索论文",
                "搜索论文",
            }

            # 去除首尾空白以及末尾常见标点，再进行精准匹配
            query = state["query"].strip().rstrip("。！？!?")
            if query not in vague_queries:
                # 返回空字典，表示不更新状态，继续后续流程
                return {}

            # 暂停图执行: 恢复执行时，用户提供的补充内容作为返回值
            topic = interrupt(
                "请补充研究主题, 列如 radar gesture recognition"
            )

            # 根据补充主题构造完整查询,明确本次检索的主题范围
            full_query = (
                f"请开始一次新的论文搜索，本次完整研究主题为: {topic}。"
                "请仅依据本次完整主题生成检索词"
            )

            # 更新查询，并将完整查询追加到消息历史中
            return {
                "query": full_query,
                "messages": [
                    *state["messages"],
                    {"role": "user", "content":full_query},
                ],
            }
        
        async def model_node(state: AgentGraphState):
            """调用模型，记录相应，并提取待执行的工具调用"""

            # 每次进入模型节点, 将模型调用轮数+1
            steps = state["steps"] + 1
            console.print(f"\n[cyan]第 {steps} 轮模型调用[/cyan]")

            # 将消息历史和可用工具定义发送给模型
            response = await self.client.responses.create(
                model=self.model,
                input=state["messages"],
                tools=self.tools,
            )

            # 复制消息历史，追加模型返回的所以输出项
            # 转化为 JSON 兼容的字典，同时省略值为 None 的字段

            messages = list(state["messages"])
            messages.extend(
                item.model_dump(mode="json",exclude_none=True)
                for item in response.output
            )

            return {
                "messages": messages,
                # 仅提取函数调用项，交给工具节点处理
                "tool_calls": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in response.output
                    if item.type == "function_call"
                ],
                # 当前相应的文本内容: 没有文本时使用空字符串
                "answer": response.output_text or "",
                "steps": steps,
            }

        async def tools_node(state: AgentGraphState):
            """依次执行工具调用, 更新业务状态，并反馈摘要准备进度"""

            messages = list(state["messages"]) 

            # 从图状态恢复业务对象，供工具执行过程读取和更新
            self.state = AgentState.from_dict(
                state["business_state"]
            )

            # 将工具调用字典还原为调用对象，逐个执行并追加工具结果
            for item in state["tool_calls"]:
                tool_call = ResponseFunctionToolCall.model_validate(item)
                messages.append(await self._execute_tool(tool_call))

            # 已设置目标论文数量时,核对并反馈最新摘要准备进度
            if self.state.target_papers > 0:
                progress = self.check_progress()
                console.print(f"摘要准备进度：{progress}")

                # 将程序核对的进度加入消息，供下一轮模型调用参考
                messages.append({
                    "role": "system",
                    "content": "Python 核对的最新摘要准备进度："
                    + json.dumps(progress, ensure_ascii=False),
                })

            answer = ""

            # 当前轮的工具执行完成后，若已经达到轮数上限，设置停止说明
            if state["steps"] >= self.max_steps:
                answer = f"达到最大执行轮数 {self.max_steps}，Agent 停止运行"

            return {
                "messages": messages,
                # 本轮工具已经执行完毕，清空待执行列表
                "tool_calls": [],
                "answer": answer,
                # 将工具执行后更新的业务对象保存回图状态
                "business_state": self.state.to_dict(),
            }

        async def finalize_node(state: AgentGraphState):
            """检查摘要证据是否齐备，记录本轮交互，并保存最终状态"""

            # 恢复最新业务状态
            self.state = AgentState.from_dict(
                state["business_state"]
            )

            answer = state["answer"]

            # 有论文数量目标时，在证据尚未备齐的回答前附加进度说明
            if self.state.target_papers > 0:
                progress = self.check_progress()
                if progress["status"] != "ready":
                    answer = (
                        f"摘要证据尚未齐备: {progress}\n\n{answer}"
                    )

            # 将本轮查询和最终回答写入业务对象的交互记忆
            self.state.remember_turn(state["query"], answer)

            return {
                "answer": answer,
                "business_state": self.state.to_dict(),
            }
        
        def after_model(state: AgentGraphState):
            """模型请求调用工具时进入工具节点，否则进入收尾节点。"""
            return "tools" if state["tool_calls"] else "finalize"
        
        def after_tools(state: AgentGraphState):
            """达到模型调用轮数上限时收尾，否则继续调用模型。"""
            if state["steps"] >= self.max_steps:
                return "finalize"
            return "model"

        # 使用 AgentGraphState 定义图中各节点共享的状态结构
        builder = StateGraph(AgentGraphState)
        
        # 注册模型、工具、澄清和收尾节点
        builder.add_node("model", model_node)
        builder.add_node("tools", tools_node)
        builder.add_node("clarify", clarify_node)
        builder.add_node("finalize", finalize_node)

        # 流程入口：先检查是否需要澄清主题，再调用模型
        builder.add_edge(START, "clarify")
        builder.add_edge("clarify", "model")

        # 根据模型是否产生工具调用，选择执行工具或直接收尾
        builder.add_conditional_edges(
            "model", 
            after_model,
            {"tools":"tools", "finalize": "finalize"}
        )

        # 工具执行后，根据轮数限制选择继续推理或收尾
        builder.add_conditional_edges(
            "tools",
            after_tools,
            {"model":"model", "finalize": "finalize"}
        )

        # 收尾完成后结束本次图执行
        builder.add_edge("finalize", END)

        # 编译状态图，并配置检查点存储以支持状态保存和中断恢复
        return builder.compile(checkpointer=self.checkpointer)

    def _graph_config(self, thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 2 * self.max_steps + 3,
        }

    async def _drive_graph(self, graph_input, config: dict) -> str:
        """执行智能体状态图，处理中断交互，并返回最终结果"""

        # 异步执行状态图，graph_input 可以是初始输入或恢复执行的命令
        # config 传入线程标识等运行配置
        # durability="sync" 表示在进入下一步前，同步完成状态持久化
        result = await self.agent_graph.ainvoke(
            graph_input,
            config=config,
            durability="sync",
        )

        # 如果执行过程中触发中断，则向用户获取补充信息
        # 恢复执行后仍可能再次中断，因此使用循环处理
        while result.get("__interrupt__"):
            # 显示中断信息，例如需要用户补充的研究主题
            console.print(result["__interrupt__"])

            # 读取用户输入，并去除首尾空白
            reply = input("补充信息: ").strip()

            # 输入为空时继续等待，不恢复状态图执行
            if not reply :
                continue

            # 将用户回复作为中断恢复值，继续执行状态图
            # 使用相同的 config, 以关联到原来的执行线程和检查点
            result = await self.agent_graph.ainvoke(
                Command(resume=reply),
                config=config,
                durability="sync",
            )

        # 执行结束后，将图中最新的业务状态还原为业务对象
        self.state = AgentState.from_dict(result["business_state"])

        # 返回状态图生成的最终答案
        return result["answer"]

    async def resume(self, thread_id: str = "main") -> str:
        """根据线程ID 恢复已有任务: 若任务已结束，则返回已保存的回答。"""

        # 根据线程 ID 构建运行配置, 用于定位该线程的检查点
        config = self._graph_config(thread_id)
        
        # 异步获取该线程最近一次保存的状态快照 
        previous = await self.agent_graph.aget_state(config)

        # 检查是否存在已保存的状态数据
        if not previous.values:
            raise ValueError("当前线程没有可恢复的任务")

        # 检查是否包含恢复任务所需的业务状态
        if "business_state" not in previous.values:
            raise ValueError("旧检查点没有完整论文状态，无法恢复该任务")

        # 从检查点恢复业务对象, 包括已保存的论文数据和任务上下文
        self.state = AgentState.from_dict(
            previous.values["business_state"]
        )

        # 没有待执行的节点，说明任务已经结束, 直接返回保存的答案
        if not previous.next:
            return previous.values.get("answer", "")

        # 传入None, 基于该线程的检查点继续执行
        # 若任务停在 interrupt(), 仍需通过 Command(resume=...) 提供恢复值
        return await self._drive_graph(None, config)
    
if __name__ == "__main__":

    import asyncio
    from pathlib import Path

    async def chat(checkpointer):
        
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
            checkpointer=checkpointer
        )

        # 加载MCP_tool
        await agent.load_mcp_tools()

        # 读取存储的state
        agent.state = AgentState.load(state_path)

        previous = await agent.agent_graph.aget_state(
            agent._graph_config("main")
        )

        if "business_state" in previous.values:
            agent.state = AgentState.from_dict(
                previous.values["business_state"]
            )

        console.print(
            f"已恢复 {len(agent.state.papers)} 篇候选论文，"
            f"{len(agent.state.paper_details)} 条详情缓存"
        )

        if previous.values:
            console.print("上次任务：", previous.values.get("query", ""))
            console.print("下一节点：", previous.next)
        else:
            console.print("尚无历史图记录")

        try:
            if previous.next:
                console.print("继续上次未完成的任务")
                answer = await agent.resume()
                agent.state.save(state_path)
                console.print(Markdown(answer))

            # 没有候选论文时，执行首次科研任务
            elif not agent.state.papers:
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

                snapshot = await agent.agent_graph.aget_state({
                    "configurable": {"thread_id":"main"},
                })

                console.print("图内模型调用次数：", snapshot.values["steps"])
                console.print("下一节点：", snapshot.next)
                agent.state.save(state_path)
                console.print(Markdown(answer))

        finally:
            agent.state.save(state_path)
    
    async def main():
        database = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "checkpoints.sqlite"
        )
        database.parent.mkdir(parents=True, exist_ok=True)

        async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
            await chat(saver)

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

Q6 model节点 和 tools节点 分别负责什么?
前者请求模型并保存输出；后者执行模型提出的工具调用并保存结果。

Q7 工具执行后为什么要回到模型节点?
模型需要读取工具结果，决定继续调用工具还是回答用户。

Q8 一次工具节点只能执行一个工具吗?
不是。一次模型响应可以包含多个工具调用，当前工具节点会逐个执行。

Q9 为什么不能只保存 response.output_text ？
工具调用等信息位于 response.output 中，后续请求需要完整的调用与结果记录。

Q10 LangGraph 执行步数是否等于模型调用次数?
不等于。工具节点也占用图的执行步数，所以项目单独使用 steps 统计模型调用次数。

"""

"""

Day 10 
Q1 检查点和之前的JSON缓存有什么区别?
检查点记录图执行过程中的状态；JSON 缓存保存你选定的论文业务数据。当前内存检查点在重启后会消失。

Q2 thread_id 是什么?
图状态记录的逻辑会话标识，同一个标识可以找到同一组当前及历史检查点

Q3 为什么在 __init__() 中创建 InMemorySaver?
让多次调用复用它。每次都创建新的 Saver，会得到新的空存储。

Q4 换一个 thread_id,是否连论文缓存也隔离了?
没有。它隔离的是图状态；当前同一个 Agent 的图外 agent.state 仍然共享。

Q5 使用同一个 thread_id, 历史消息就会自动追加?
不会。状态如何更新取决于字段规则和本次输入；当前完整传入的消息列表会覆盖原值。

Q6 SQLite 相比内存检查点增加了什么能力?
将图状态保存到文件，程序重启后仍可读取

Q7 为什么项目使用 AsyncSqliteSaver ？
项目通过 ainvoke()、aget_state() 等异步接口运行，需要使用异步 Saver。

Q8 为什么数据库连接要包住整个交互过程?
图执行和读取检查点时都需要使用连接，交互结束后再关闭。

Q9 恢复时只保持 thread_id 相同就够了吗？
还需要读取同一个数据库，并使用匹配的图定义。

Q10 SQLite 会自动保存图外的 agent.state 吗?
不会。检查点保存的是图状态；当前论文业务缓存仍由 JSON 保存逻辑管理。
"""

"""

Day 11

Q1 为什么接入 SQLite 后，还需要修改论文状态?
SQLite 保存图状态；原来的论文、摘要和计划在图外的 self.state 中，需要主动放入检查点。

Q2 to_dict() 和 from_dict() 分别做什么?
前者把完整业务状态转换成可保存的字典；后者从字典恢复 AgentState 对象.

Q3 为什么恢复任务不能再次调用 run() ?
run() 会初始化新任务。恢复任务应读取原检查点，用 None 继续执行；回答中断问题时用 Command(resume=...)。

Q4 为什么增加 finalize 节点?
让最终回答、证据检查结果和聊天历史一起进入检查点。

Q5 恢复后回从退出时的那一行继续吗?
不会。恢复按节点进行，未完成的节点可能重新执行。

"""