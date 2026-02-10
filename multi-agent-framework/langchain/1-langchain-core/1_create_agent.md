# create agent 作为目前，langchain所推荐的创建agent的方式

```python
from langchain.agents import create_agent
# 示例：创建 Agent
agent = create_agent(llm, tools, prompt)
```

### 方法定义
Creates an agent graph that calls tools in a loop until a stopping condition is met.

基本上也就是 Agent 最形象的定义，`一个无限循环,直到满足某个条件的工具`



### 参数说明
| 参数名 | 类型 | 必须 | 说明 |
| :--- | :--- | :--- | :--- |
| **`model`** | `BaseChatModel` | 是 | 驱动 Agent 的大语言模型实例（支持 Tool Calling 的模型）。 |
| **`tools`** | `Sequence[BaseTool]` | 否 | Agent 可以调用的工具列表，通常是自定义函数或预置工具。 |
| **`middleware`** | `Sequence[AgentMiddleware[StateT_co, ContextT]]` | 是 | 定义 Agent 行为的提示词模版，需包含特定的占位符（如 `agent_scratchpad`）。 |

> [!NOTE] 核心概念解析
>
> **model (基础的模型)**
> 这里的 `model` 是 Agent 运行时的核心， 模型的选择直接决定了 Agent 的能力和行为。
> **类型：BaseChatModel**
> ```python
> from langchain_core.language_models.chat_models import BaseChatModel
> class BaseChatModel(BaseLanguageMode[AIMessage], ABC):
> 这里是定义了 BaseChatModel 的基类，所有模型的行为，方法均来源于此
> ainvoke（这里只讲异步的核心方法）
>
>
>
>
>
>
>
>
>
>
>
> ```
> **调用逻辑**
> 




> **Middleware (中间件)**  
> 这里的 `middleware` 是 Agent 运行时的核心拦截器，它不仅像 Prompt 那样定义行为，更是一个**函数序列**。  
> 它可以：
> 1. 在 LLM 调用前修改 State（例如注入上下文）。
> 2. 在 LLM 调用后处理 Output（例如格式校验）。
> 3. 控制 Agent 的执行流（例如中断或重试）。

