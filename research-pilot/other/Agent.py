# import os
# from openai import OpenAI
# from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# load_dotenv()
# console = Console()

# client = OpenAI(
#     api_key=os.environ["OPENAI_API_KEY"],
#     base_url=os.environ["OPENAI_BASE_URL"]
# )

# model=os.environ["OPENAI_MODEL"]
# """ OpenAI example """


# response = client.responses.create(

#     model=model,
#     input="Write a one-sentence bedtime story about a unicorn.",

# )

# print(response.output)
# print()
# print(response.output_text)

"""Prompt engineering"""

# response = client.responses.create(
#     model=model,
#     reasoning={"effort": "low"},
#     instructions="Talk like a pirate.",
#     input="Are semicolons optional in JavaScript?",

# )

# console.print(Panel("原始 Response Output", expand=False))
# console.print(response.output)
# console.print("\n" + "=" * 40 + "\n")
# console.print(Markdown(response.output_text))

"Structured model outputs"

# from openai import OpenAI
# from pydantic import BaseModel

# class CalendarEvent(BaseModel):
#     name: str
#     date: str
#     participants: list[str]

# response = client.responses.parse(
#     model=model,
#     input=[
#         {"role": "system", "content": "Extract the event information."},
#         {
#             "role": "user",
#             "content": "Alice and Bob are going to a science fair on Friday.",
#         },
#     ],
#     text_format=CalendarEvent,
# )


# console.print(Panel("原始 Response Output", expand=False))
# console.print(response.output)
# # console.print("\n" + "=" * 40 + "\n")
# console.print(Panel("结构化输出", expand=False))
# console.print(response.output_parsed)


# from openai import OpenAI
# from pydantic import BaseModel


# class Step(BaseModel):
#     explanation: str
#     output: str


# class MathReasoning(BaseModel):
#     steps: list[Step]
#     final_answer: str


# response = client.responses.parse(
#     model=model,
#     input=[
#         {
#             "role": "system",
#             "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
#         },
#         {"role": "user", "content": "how can I solve 8x + 7 = -23"},
#     ],
#     text_format=MathReasoning,
# )

# math_reasoning = response.output_parsed

# console.print(Panel("结构化输出",expand=False))
# console.print(math_reasoning)


# from openai import OpenAI
# from pydantic import BaseModel

# class EntitiesModel(BaseModel):
#     attributes: list[str]
#     colors: list[str]
#     animals: list[str]


# with client.responses.stream(
#     model=model,
#     input=[
#         {"role": "system", "content": "Extract entities from the input text"},
#         {
#             "role": "user",
#             "content": "The quick brown fox jumps over the lazy dog with piercing blue eyes",
#         },
#     ],
#     text_format=EntitiesModel,
# ) as stream:
#     for event in stream:
#         if event.type == "response.refusal.delta":
#             print(event.delta, end="")
#         elif event.type == "response.output_text.delta":
#             print(event.delta, end="", flush=True)
#         elif event.type == "response.error":
#             print(event.error, end="",flush=True)
#         elif event.type == "response.completed":
#             print("Completed")  # print(event.response.output)

#     # final_response = stream.get_final_response()
#     # print(final_response)


# """Function Calling"""

# from openai import OpenAI
# import json

# # 定义可以调用的函数
# tools = [
#     {
#         "type": "function",
#         "name": "get_horoscope",
#         "description": "Get today's horoscope for an astrological sign.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "sign":{
#                     "type": "string",
#                     "description": "An astrological sign like Taurus or Aquarius",
#                 },
#             },
#             "required":["sign"],
#         },
#     },
# ]

# def get_horoscope(sign):
#     return f"{sign}: Next Tuesday you will befriend a baby otter."


# # Create a running input list we will add to over time
# input_list = [{"role": "user", "content": "What is my horoscope? I am an Aquarius."}]

# # 2. Prompt the model with tools defined
# response = client.responses.create(
#     model=model,
#     tools=tools,
#     input=input_list,
# )

# # Save function call outputs for subsequent requests
# input_list += response.output

# # console.print(json.loads(response.output[1].arguments)['sign'])

# for item in response.output:
#     if item.type == "function_call":
#         if item.name == "get_horoscope":
#             # 3. Execute the function logic for get_horoscope
#             sign = json.loads(item.arguments)["sign"]
#             horoscope = get_horoscope(sign)

#             # 4. Provide function call results to the model
#             input_list.append(
#                 {
#                     "type": "function_call_output",
#                     "call_id": item.call_id,
#                     "output": horoscope,
#                 }
#             )

# console.print("Final input:")
# console.print(input_list)

# response = client.responses.create(

#     model=model,
#     instructions="Respond only with a horoscope generated by a tool.",
#     tools=tools,
#     input=input_list
# )

# # 5. The model should be able to give a response!
# print("Final output:")
# print(response.model_dump_json(indent=2))
# print("\n" + response.output_text)


import os
import json
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from app.tools.paper_search import search_papers


load_dotenv()
console = Console()
client = OpenAI(
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
            "keyword": {"type": "string", "description": "论文检索关键词"},
            "start_year": {"type": "integer", "default": 2023},
            "end_year": {"type": "integer", "default": 2026},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
        },
        "required": ["keyword"],
        "additionalProperties": False,
    },
    "strict": False,
}

# 工具分发表
TOOL_REGISTRY = {"search_papers": search_papers}


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
        """执行工具调用，并构造工具结果消息"""

        # 1. 解析工具参数
        args = json.loads(tool_call.arguments)

        # 2. 查找工具
        tool_func = TOOL_REGISTRY[tool_call.name]

        # 3. 执行异步工具
        result = await tool_func(**args)

        # 4. Pydantic 转换为字典
        result_data = [paper.model_dump(mode="json") for paper in result]

        # 5. 构造工具结果消息
        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(result_data, ensure_ascii=False),
        }

    def __init__(self, client: AsyncOpenAI, model: str, max_steps: int = 5):

        self.client = client
        self.model = model
        self.max_steps = max_steps

    async def run(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]

        # Agent Loop
        for step in range(self.max_steps):
            console.print(f"\n[cyan]第{step + 1} 轮模型调用[/cyan]")

            # 1. 询问模型
            responses = client.responses.create(
                model=model, input=messages, tools=[SEARCH_PAPERS_TOOL]
            )

            # console.print(responses.output)
            # messages += responses.output

            # 2. 保存模型输出
            messages.extend(
                item.model_dump(exclude_none=True) for item in responses.output
            )

            # 3. 提取工具调用
            tool_calls = [
                item for item in responses.output if item.type == "function_call"
            ]
            # console.print(tool_calls)

            # 4. 如果模型不在调用工具，说明任务完成
            if not tool_calls:
                # console.print("\n[green]最终回答[/green]")
                # console.print(Markdown(responses.output_text or ""))
                return responses.output_text or ""

            # 5. 执行工具
            for tool_call in tool_calls:
                tool_output = await self._execute_tool(tool_call)

                # 6. 将工具结果加入消息历史
                messages.append(tool_output)

            # # 5.将工具结果返回模型
            # final_response = client.responses.create(
            #     model=model,
            #     input=messages,
            #     tools=[SEARCH_PAPERS_TOOL]
            # )
        return f"[yellow]达到最大执行轮数 {self.max_steps}, Agent 停止运行[/yellow]"
        # console.print("[yellow]达到最大执行轮数, Agent 停止运行[/yellow]")

    async def _execute_tool(self, tool_call) -> dict:
        """执行工具调用,并构造工具结果消息"""

        # 1. 解析工具参数
        args = json.loads(tool_call.arguments)

        # 2. 查找工具
        tool_func = TOOL_REGISTRY[tool_call.name]

        result = await tool_func(**args)

        # console.print(result)

        # 如果Paper 是 Pydantic 模型
        result_data = [paper.model_dump(mode="json") for paper in result]

        return {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(result_data, ensure_ascii=False),
        }


# async def main():
#     messages = [
#         {
#             "role": "user",
#             "content": "搜索2024到2026年毫米波雷达手势识别论文5篇"
#         }
#         ]
#     max_steps = 5
#     for step in range(max_steps):

#         console.print(f"\n[cyan]第{step + 1} 轮模型调用[/cyan]")

#         # 1. 询问模型
#         responses = client.responses.create(
#             model= model,
#             input= messages,
#             tools= [SEARCH_PAPERS_TOOL]
#         )

#         # console.print(responses.output)
#         # messages += responses.output

#         # 2. 保存模型输出
#         messages.extend(
#             item.model_dump(exclude_none=True)
#             for item in responses.output
#         )

#         # 3. 提取工具调用
#         tool_calls = [
#             item
#             for item in responses.output
#             if item.type == "function_call"
#         ]
#         # console.print(tool_calls)

#         # 4. 如果模型不在调用工具，说明任务完成
#         if not tool_calls:
#             console.print("\n[green]最终回答[/green]")
#             console.print(Markdown(responses.output_text or ""))
#             return

#         # 5. 执行工具
#         for tool_call in tool_calls:

#             args = json.loads(tool_call.arguments)
#             # console.print("Tool name: ", tool_call.name)
#             # console.print("Tool parameters: ", args)

#             tool_func = TOOL_REGISTRY[tool_call.name]

#             result = await tool_func(**args)

#             # console.print(result)

#             # 如果Paper 是 Pydantic 模型
#             result_data = [
#                 paper.model_dump(mode="json")
#                 for paper in result
#             ]

#             # 6. 将工具结果加入消息历史
#             messages.append({
#                 "type": "function_call_output",
#                 "call_id": tool_call.call_id,
#                 "output":json.dumps(
#                     result_data,
#                     ensure_ascii=False
#                 )
#             })

#         # # 5.将工具结果返回模型
#         # final_response = client.responses.create(
#         #     model=model,
#         #     input=messages,
#         #     tools=[SEARCH_PAPERS_TOOL]
#         # )

#     console.print("[yellow]达到最大执行轮数, Agent 停止运行[/yellow]")


async def main():

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"]
    )

    agent = PaperSearchAgent(
        client=client, model=os.environ["OPENAI_MODEL"], max_steps=5
    )

    result = await agent.run("搜索2024到2026年有关agent的高影响力会议论文5篇")

    console.print("\n[green]最终答案[/green]")
    console.print(Markdown(result))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

# class PaperSearchAgent:

#     def __init__(self, client: AsyncOpenAI, model: str):
#         #TODO
#         self.client = client
#         self.model = model

#     async def run(self, query:str) -> str:
#         #TODO
#         messages = [{"role": "user", "content": query}]

#         # 1. 询问模型
#         response = await self.client.responses.create(
#             model=self.model,
#             input=messages,
#             tools=[SEARCH_PAPERS_TOOL]
#         )
#         msg = response.choices[0].message

#         # 如果无需工具，直接返回内容
#         if not msg.tool_calls:
#             return msg.content or ""

#         # 2. 存在工具调用，必须先计入历史
#         messages.append(msg)
#         tool_call = msg.tool_calls[0]
#         args = json.loads(tool_call.function.arguments)
