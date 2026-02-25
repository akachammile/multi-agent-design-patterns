import asyncio
from langchain_core.runnables.base import RunnableEach
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = ChatPromptTemplate.from_template("Tell me a short joke about{topic}")
model = ChatOpenAI(
    model="qwen3.5-flash-2026-02-23",
    api_key="sk-b86e4fd521754a80b0210a8eab6f4d9c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
output_parser = StrOutputParser()
runnable = prompt | model | output_parser
runnable_each = RunnableEach(bound=runnable)


async def main():
    output = await runnable_each.ainvoke(
        [{"topic": "Computer Science"}, {"topic": "Art"}, {"topic": "Biology"}]
    )
    print(output)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
