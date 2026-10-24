import pytest

"""
learn how to use pytest

1.pytest.mark.parametrize


"""
# 对内置函数进行参数化处理
@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected
    