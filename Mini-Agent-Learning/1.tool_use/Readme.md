# 工具调用：定义并执行自定义工具

本章通过一个可复用的工具抽象和四则运算计算器，学习如何把普通 Python 函数组织成智能体可以描述、调用并读取结果的工具。

## 学习目标

- 使用抽象基类约定工具必须提供的名称、描述、参数和执行方法。
- 用 JSON Schema 描述工具参数，并转换成 OpenAI function tool schema。
- 使用异步 `execute` 方法执行工具，通过统一的结果类型返回成功或错误。
- 理解工具定义与模型选择、应用程序实际执行工具之间的分工。

## 目录结构

```text
Tools/
├── base.py             # Tool 抽象类和 ToolResult
└── CalculatorTool.py   # 四则运算工具及自测
```

## 核心设计

### `ToolResult`：统一执行结果

`Tools/base.py` 中的 `ToolResult` 是一个 Pydantic 模型：

| 字段 | 含义 |
| --- | --- |
| `success` | 本次执行是否成功 |
| `content` | 成功结果，默认为空字符串 |
| `error` | 错误说明，默认为 `None` |

成功时通常返回 `ToolResult(success=True, content="...")`；失败时返回 `ToolResult(success=False, error="...")`。统一结果结构可以让调用方以相同方式处理不同工具。

### `Tool`：工具接口

`Tool` 是抽象基类，子类需要实现以下成员：

- `name`：供调用方识别工具的名称。
- `description`：说明工具用途，帮助模型判断何时调用。
- `parameters`：JSON Schema 格式的参数定义。
- `execute(...)`：异步执行工具逻辑，并返回 `ToolResult`。

`to_openai_schema()` 会把名称、描述和参数组合成 function tool 定义。它只负责生成工具描述；模型如何选择工具、应用程序如何分派调用，以及如何把结果交回模型，都需要由外层智能体流程负责。

### `CalculatorTool`：实现四则运算

计算器继承 `Tool`，接受两个数字 `a`、`b` 和一个运算符 `operator`：

| 运算符 | 操作 |
| --- | --- |
| `add` | 加法 |
| `sub` | 减法 |
| `mul` | 乘法 |
| `div` | 除法 |

参数 schema 将 `operator` 限制在上述枚举值中。执行时，工具会返回运算结果；除数为零或运算符未知时，会返回 `success=False` 和错误信息。

> 参数 schema 是提供给工具调用方的接口描述，不会自动验证 Python 方法收到的参数。实际项目中应由调用工具的代码负责校验不可信或不符合 schema 的输入。

## 运行示例

在项目根目录执行计算器内置的自测：

```bash
cd Mini-Agent-Learning/1.tool_use/Tools
python CalculatorTool.py
```

自测覆盖四种运算、除数为零和非法运算符。通过时会逐项输出 `PASS`。

`base.py` 也包含一个简单的 `AddTool` 示例，用于查看工具属性、生成 OpenAI schema 并执行加法：

```bash
cd Mini-Agent-Learning/1.tool_use/Tools
python base.py
```

本章使用 Pydantic 2 定义结果模型（例如 `model_dump()` 和 `model_json_schema()`）；运行 `base.py` 演示还需要 `rich`。若当前 Python 环境缺少依赖，请安装 `pydantic>=2`，并在运行 `base.py` 时安装 `rich`。

## 手动调用计算器

```python
import asyncio
from CalculatorTool import CalculatorTool


async def main():
    calculator = CalculatorTool()
    result = await calculator.execute(a=10, b=2, operator="div")
    print(result.model_dump())
    print(calculator.to_openai_schema())


asyncio.run(main())
```

调用 `execute` 时使用关键字参数，并通过 `await` 等待异步执行。示例结果为：

```text
{'success': True, 'content': '5.0', 'error': None}
```

## 本章流程回顾

1. 继承 `Tool`，实现工具的元信息、参数 schema 和执行逻辑。
2. 将 `parameters` 与名称、描述转换为模型可理解的工具定义。
3. 由外层应用根据模型返回的工具名和参数调用 `execute`。
4. 检查 `ToolResult.success`，并将结果或错误交由外层流程处理。