from langchain.tools import tool


@tool
def add(a: int, b: int) -> int:
    """
    tool for calculate a add b
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """
    tool for calculate a add b
    """
    return a * b
