from base import ToolResult, Tool
from typing import Any

class CalculatorTool(Tool):

    @property
    def name(self) -> str:
        return "calculator" 

    @property
    def description(self) -> str:
        return "a simple calculator, can operate add, substract, multiply, divide" 

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "the first number"
                },
                "b": {
                    "type": "number",
                    "description": "the second number"
                },
                "operator": {
                    "type": "string",
                    "enum": ["add", "sub", "mul", "div"],
                    "description": "The arithmetic operation to perform: add, sub, mul, or div.",
                }
            },
            "required":["a", "b", "operator"],
            "additionalProperties":False
        }

    async def execute(self, a:float, b:float, operator: str = "add") -> ToolResult:
            content = ""
            error = ""
            if operator == "add":
                content = a + b 
            elif operator == "sub":
                content = a - b
            elif operator == "mul":
                content = a * b
            elif operator == "div":

                if b == 0:
                    error = "b can not be zero !"
                    return ToolResult(
                        success=False,
                        error=error
                    ) 
                content = a / b
            else:
                error =  f"Unknown operator: ‘{operator}', must be one of add, sub, mul, div"
                return ToolResult(
                    success=False,
                    error=error
                )
            return ToolResult(
                success=True,
                content=str(content)
            )
if __name__ == "__main__":
    import asyncio
    import math

    async def test_calculator():
        tool = CalculatorTool()

        # 测试用例
        test_cases = [
            (3.0, 5.0, "add", 8.0),
            (3.0, 5.0, "sub", -2.0),
            (3.0, 5.0, "mul", 15.0),
            (10.0, 2.0, "div", 5.0),
        ]

        print("========== Calculator Tests ==========")

        # 测试四则运算
        for a, b, operator, expected in test_cases:

            result = await tool.execute(
                a=a,
                b=b,
                operator=operator
            )

            assert result.success is True
            assert math.isclose(float(result.content), expected)

            print(
                f"PASS: {a} {operator} {b} = {result.content}"
            )

        # 测试除数为零
        result = await tool.execute(
            a=10.0,
            b=0.0,
            operator="div"
        )

        assert result.success is False
        assert result.error is not None

        print(f"PASS: Division by zero -> {result.error}")

        # 测试非法运算符
        result = await tool.execute(
            a=10.0,
            b=5.0,
            operator="power"
        )

        assert result.success is False
        assert result.error is not None

        print(f"PASS: Invalid operator -> {result.error}")

        print("========== All Tests Passed ==========")


    if __name__ == "__main__":
        asyncio.run(test_calculator())
            
                 
