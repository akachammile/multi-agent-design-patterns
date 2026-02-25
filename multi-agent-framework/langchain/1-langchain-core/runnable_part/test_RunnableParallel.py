from langchain_core.runnables import RunnableLambda


def add_one(x: int) -> int:
    return x + 1


def mul_two(x: int) -> int:
    return x * 2


def mul_three(x: int) -> int:
    return x * 3


runnable_1 = RunnableLambda(add_one)
runnable_2 = RunnableLambda(mul_two)
runnable_3 = RunnableLambda(mul_three)
