import os
import json
import inspect
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.tools.paper_search import search_papers
from app.tools.paper_rank import rank_papers
from app.tools.paper_detail import get_paper_detail
from app.agent.state import AgentState
from rich.console import Console
from rich.markdown import Markdown

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
        "parameters": {
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

PAPER_DETAIL_TOOL = {
    "type": "function",
    "name": "get_paper_detail",
    "description": "通过 OpeAlex Work ID 获取论文摘要和详细元数据。",
    "parameters": {
        "type": "object",
        "properties": {
            "paper_id": {
                "type":"string",
                "description": "OpenAlex Work ID，例如 W123456789。",
            },
        },
        "required": ["paper_id"],
        "additionalProperties": False,
    },
    "strict": False
}

TOOLS = [
    SEARCH_PAPERS_TOOL,
    RANK_PAPERS_TOOL,
    PAPER_DETAIL_TOOL,
]



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
        }

    async def run(self, query: str) -> str:

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
                ),
            },
            {"role": "user", "content": query}
            ]

        # Agent Loop
        for step in range(self.max_steps):
            console.print(f"\n[cyan]第 {step + 1} 轮模型调用[/cyan]")

            # 1. 询问模型
            responses = await self.client.responses.create(
                model=self.model, input=messages, tools=TOOLS
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
                return responses.output_text or ""

            # 5. 执行工具
            for tool_call in tool_calls:
                tool_output = await self._execute_tool(tool_call)

                # 6. 将工具结果加入消息历史
                messages.append(tool_output)

        return f"达到最大执行轮数 {self.max_steps}，Agent 停止运行"
    
    async def _execute_tool(self, tool_call) -> dict:
        """执行工具"""
        try:

            if tool_call.name not in self.tool_registry:
                raise ValueError(f"未注册的工具: {tool_call.name}")

            # json -> python dict
            args = json.loads(tool_call.arguments)

            if not isinstance(args, dict):
                raise ValueError("工具参数必须是 JSON 对象")

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

        console.print(f"[dim]请求 OpenAlex 详情：{work_id}[/dim]")

        detail = await get_paper_detail(paper_id)

        self.state.paper_details[paper_url] = detail

        return detail

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
    

if __name__ == "__main__":

    # def sync_func():
    #     return [1, 2, 3]

    # async def async_func():
    #     return [1, 2, 3]

    # a = sync_func()
    # b = async_func()

    # console.print(inspect.isawaitable(a))
    # console.print(inspect.isawaitable(b))

    import asyncio

    async def test():
        agent = PaperSearchAgent(
            client=client,
            model=model,
        )

    #     query = (
    #         "请按以下步骤完成任务："
    #         "1. 调用 search_papers，搜索 2023—2026 年关于 machine learning 的论文，返回 5 篇。"
    #         "2. 调用 rank_papers，按引用量选出其中前 2 篇，保留论文的 url。"
    #         "3. 根据选中论文的 url，分别调用 get_paper_detail 获取摘要；"
    #         "如果支持，请在同一轮中提出这两个工具调用。"
    #         "4. 用中文列出两篇论文的标题、年份、引用量、链接，并分别总结摘要。"
    #         "如果没有摘要，请明确说明。"
    #     )

    #     result = await agent.run(query)

    #     console.print("\n[green]最终回答：[/green]")
    #     console.print(Markdown(result), markup=False)
    # asyncio.run(test())
        answer = await agent.run(
            "搜索 2024—2026 年机器学习的论文，"
            "从搜索结果中选出引用量最高的 3 篇。"
        )
        console.print(Markdown(answer))
        answer = await agent.run(
            "刚才选出的第一篇论文是什么？其摘要主要讲了什么？"
                    
        )
        console.print(Markdown(answer))

    asyncio.run(test())
    pass



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
