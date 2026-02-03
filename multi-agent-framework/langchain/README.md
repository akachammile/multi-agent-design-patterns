# LangChain (2025 Edition)

## 🌟 2025 年的 10 月 LangChain 正式发布了其V1.0的版本, 对其内容和架构做出了巨大的更新

截止 2025 年，LangChain 已从最初的大杂烩式工具集，进化为一个专注于 **构建可适应生态进化的 Agent 平台**。
其核心理念已从“链（Chain）”全面升级为“智能体架构（Agent Architecture）”。

### � 核心更新：`create_agent` 的回归与统一

LangChain 最显著的变化是将底层使用langGraph重新设计, 
功能上: 引入了统一的入口函数 `create_agent`, 并且加入了 `Middleware` 等特性. 
该部分代码将会从底层对 Langchain 做出彻底的解读.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-5-20250929",  # 支持直接指定模型 ID 或模型对象
    tools=[get_weather],                  # 工具列表
    middlewares=[example_middleware],
    system_prompt="You are a helpful assistant",
)
```

这个新的抽象层实际上构建在 **LangGraph** 之上，融合了易用性与灵活性：
- **极简入门**：10 行代码即可启动一个具备工具调用能力的 Agent。
- **深度定制**：通过 Middleware 和底层 Graph 访问，支持复杂的上下文工程（Context Engineering）。

### 🏗️ 架构分层

新版架构更加模块化，清晰地分为以下层次：

#### 1. 基础抽象层 (`langchain-core`) - 标准化接口
这是整个生态的基石，定义了所有组件交互的标准协议。
- **Standard Model Interface**: 无论底层是 OpenAI、Anthropic 还是本地模型，LangChain 提供了统一的调用接口，可以在不修改业务逻辑的情况下无缝切换供应商。
- **Runnable Protocol**: 所有组件（Prompts, Models, Parsers）都遵循 Runnable 协议，支持统一的 `invoke`, `stream`, `batch` 操作。

#### 2. 编排引擎 (`LangGraph`) - 智能体的“大脑”
LangChain 的 Agent 现在默认基于 **LangGraph** 构建。
- **持久化执行 (Durable Execution)**: 支持长运行任务的中断与恢复。
- **人机交互 (Human-in-the-loop)**: 原生支持人工审核、修改状态等操作。
- **循环与状态管理**: 专为 Agent 的循环思考模式（Plan-Execute-Refine）设计，解决了 DAG（有向无环图）无法处理复杂决策的问题。

#### 3. 应用层 (`langchain`) - 高级封装
- **`create_agent`**: 针对最常见的 Agent 模式（如 ReAct, Tool Calling）提供的高级封装。
- **Integrations**: 提供了数百种预构建的工具和模型集成。