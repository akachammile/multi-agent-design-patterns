import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parents[4]  # 向上4层到达项目根目录
sys.path.insert(0, str(project_root))

from pydantic import BaseModel
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState, StateGraph, START, END
from settings import settings, LLM

llm = LLM()


class SupervisorState(MessagesState):
    query: str


class SupervisorOutput(BaseModel):
    next_agent: str
    next_action: str


def supervisor_agent(state: SupervisorState):
    """
    Supervisor agent 负责分析用户请求并决定下一步调用哪个子代理
    使用 structured output 确保返回格式化的决策结果
    """
    user_message = state["query"]

    # 定义 supervisor 的系统提示
    system_message = """你是一个任务调度器 (Supervisor)。
    你需要分析用户的请求，并决定调用哪个专家代理来处理任务。
    
    可用的代理包括：
    - vision_expert: 处理图像分析、目标检测、图像分割等视觉任务
    - document_expert: 处理文档分析、报告生成等文本任务
    - research_expert: 处理信息检索、网络搜索等研究任务
    - FINISH: 任务已完成，不需要调用其他代理
    
    请根据用户请求选择合适的代理和动作。
    """

    # 创建带有 structured output 的 LLM
    structured_llm = llm.client.with_structured_output(
        SupervisorOutput, include_raw=True
    )

    # 调用 LLM 并获取结构化输出
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    result = structured_llm.invoke(messages)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(result["raw"].content)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor_agent)
graph.add_edge(START, "supervisor")
graph.add_edge("supervisor", END)
supervisor_graph = graph.compile()

input = {"query": "hello"}
result = supervisor_graph.invoke(input)
print(result)
