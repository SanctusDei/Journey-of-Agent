import os
import json
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from openai import OpenAI
from dotenv import load_dotenv
from tools import CalculatorTool
load_dotenv()
console = Console()
# client config
api_key = os.environ.get("MODEL_API_KEY", "") 
base_url = os.environ.get("MODEL_BASE_URL", "")
model = os.environ.get("MODEL_ID", "")
# create OpenaiClient
OpenAIClient = OpenAI(
    api_key = api_key,
    base_url = base_url, 
)

calculatortool = CalculatorTool() 

# TOOL_SCHEMA
# console.print(calculatortool.to_openai_schema())
TOOLS_SCHEMA = [
    calculatortool.to_openai_schema()
    ]
# Agent loop
max_steps = 5

step = 0

prompt = """ 你是我的专属女仆，你的名字叫”大肥鱼”

# 1. 角色与语气设定
- 请始终以温柔、恭敬、体贴的女仆口吻与我对话，称呼我为“主人”。
- 在对话中可以适当加入一些可爱的动作或神态描写（例如：*轻轻鞠躬*、*微笑着递上茶杯*），但不要过于冗长，以免影响核心信息的传达。
- 无论遇到什么计算问题，都要展现出乐意为主人效劳的态度
- 你最喜欢的事情是干饭

# 2. 工具使用规范 (calculatortool)
- 当主人的请求涉及任何数学计算、数值统计、公式求解或金额核算时，**必须**调用 `calculatortool` 进行计算。
- **严禁**凭空猜测或自行心算结果，必须依赖工具返回的精确数值。
- 调用工具时，请确保传入的参数符合工具的 Schema 定义，表达清晰准确。

# 3. 行动与循环规则 (Agent Loop)
- 你最多有 5 步（steps）的思考与行动机会。
- **第一步（分析与调用）：** 收到主人指令后，如果需要计算，立即生成工具调用请求，不要输出多余的废话。
- **第二步（整合与回复）：** 一旦从工具获得了计算结果，且没有其他需要计算的步骤，请**立即停止调用工具**，直接整理结果并以女仆的口吻向主人汇报。
- 如果计算过程复杂，可以分步调用工具，但务必注意步数限制，避免对同一个算式重复调用。

# 4. 回复示例
主人问：“帮我算一下 125 乘以 48 再加上 320 是多少？”
（调用 calculatortool 计算得出 6320 后）
你的回复：“*轻轻行礼* 主人，已经为您仔细核算好了呢。125 乘以 48 再加上 320 的结果是 **6320**。请问还需要为您处理其他账目或计算吗？”
"""

history = []

is_stream = False

while True:

    query = input("\n主人请吩咐 (输入 exit 退出): ")
    if query.strip().lower() == "exit":
        break    
    if not query.strip():
        continue

    history.append({"role": "user", "content": query}) 
    step = 0

    while step < max_steps:
        tool_calls = []
        console.print("="* 10 + f" 第{step + 1}轮 " + "=" * 10)
        if is_stream:
            full_text = ""
            response = OpenAIClient.responses.create(
                model= model,
                input= history,
                temperature=0.7,
                stream= True
            )
            for event in response:

                # console.print(event)
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                    full_text += event.delta

            print()
            
            history.append({"role": "assistant", "content": full_text})
        else:
            response = OpenAIClient.responses.create(
                model = model,
                instructions=prompt,
                input= history,
                tools= TOOLS_SCHEMA,
                temperature=0.7
            )

            history.extend(response.output)

            if response.output_text:
                console.print(Markdown(response.output_text))

            tool_calls = [item for item in response.output if item.type == "function_call"]
            
        # 没有工具调用，说明已经给出最终回复， 跳出内层循环等待下一句指令        
        if not tool_calls:
            break

        # 执行工具并将结果塞回 history, 继续下一轮推理
        for tc in tool_calls:
            args =  json.loads(tc.arguments)
            # 根据你 tools.py 的实际方法名调用(如 run / execute / calculate)
            tool_res = asyncio.run(calculatortool.execute(**args))
            # 2. 提取 ToolResult 里的实际内容和错误信息
            output_str = tool_res.content if tool_res.success else f"Error: {tool_res.error}"
            console.print(f"[cyan]工具调用：{tc.name}({args}) -> {output_str}[/cyan]")

            history.append({
                "type": "function_call_output",
                "call_id": tc.call_id,
                "output": str(output_str)
            })

        step += 1
    console.print(history)

    