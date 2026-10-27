import pytest

"""
learn how to use pytest

1.pytest.mark.parametrize


"""
# 对内置函数进行参数化处理
# @pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
# def test_eval(test_input, expected):
#     assert eval(test_input) == expected



# AsyncMock 模拟异步函数

import asyncio
from unittest.mock import AsyncMock

def test_mock_search():
	mock_search = AsyncMock(return_value=["paper1", "paper2"])

	async def run_search():
		result = await mock_search(keyword="mmWave")
		assert result == ["paper1", "paper2"]

	asyncio.run(run_search())
	mock_search.assert_awaited_once()

if __name__ == "__main__":
	test_mock_search()