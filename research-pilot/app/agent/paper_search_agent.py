import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from ..tools import paper_search
from rich.console import Console

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

# 工具分发表
TOOL_REGISTRY = {

    "search_papers": paper_search,

}


class PaperSearchAgent:
    def __init__(self, client: AsyncOpenAI, model: str, max_steps: int = 5):
        self.client = client
        self.model = model
        self.max_steps = max_steps

    async def run(self, query: str) -> str:

        messages = [{"role": "user", "content": query}]

        # Agent Loop
        for step in range(self.max_steps):
            console.print(f"\n[cyan]第 {step + 1} 轮模型调用[/cyan]")

            # 1. 询问模型
            responses = await self.client.responses.create(
                model=self.model, input=messages, tools=[SEARCH_PAPERS_TOOL]
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
        try:
            if tool_call.name not in TOOL_REGISTRY:
                raise ValueError(f"未注册的工具: {tool_call.name}")

            args = json.loads(tool_call.arguments)

            if not isinstance(args, dict):
                raise ValueError("工具参数必须是 JSON 对象")

            tool_func = TOOL_REGISTRY[tool_call.name]
            result = await tool_func(**args)

            result_data = [
                paper.model_dump(mode="json")
                for paper in result
            ]

            output = json.dumps(result_data, ensure_ascii=False)

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
