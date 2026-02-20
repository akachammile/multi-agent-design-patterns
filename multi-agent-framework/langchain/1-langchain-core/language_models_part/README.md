# Language Models — 语言模型抽象层

> 本模块定义了 LangChain 中所有语言模型的标准接口。
> 无论是 OpenAI、Anthropic、Qwen 还是本地模型，都必须遵循这套协议。

---

## 🏗️ 继承链

```
Runnable[Input, Output]                  ← 万物基类（统一调用协议）
    │
RunnableSerializable[Input, Output]      ← 加入序列化能力
    │   └── 继承了 Serializable（存盘/读盘）
    │   └── 继承了 Runnable（invoke/stream/batch）
    │
BaseLanguageModel[LanguageModelOutputVar] ← 语言模型基类（泛型：输出类型）
    │
    ├── BaseLLM                           ← 文本补全模型（输入输出都是 str）
    │
    └── BaseChatModel                     ← 聊天模型（输入消息列表，输出 AIMessage）
        │
        └── BaseChatOpenAI                ← OpenAI 聊天模型
            │
            ├── ChatOpenAI                ← GPT-4 等
            └── AzureChatOpenAI           ← Azure 部署
```

---

## 🔑 核心概念

### 1. `BaseLanguageModel` — 所有 LLM 的祖先

```python
class BaseLanguageModel(
    RunnableSerializable[LanguageModelInput, LanguageModelOutputVar], ABC
):
```

- `LanguageModelInput` — 输入类型，支持 `str`、`list[BaseMessage]`、`PromptValue`
- `LanguageModelOutputVar` — 输出类型，被约束为 `AIMessage` 或 `str`

### 2. `BaseChatModel` — 聊天模型基类

```python
class BaseChatModel(BaseLanguageModel[AIMessage], ABC):
```

- 将输出类型固定为 `AIMessage`
- 子类**必须实现** `_generate()` 方法（核心推理逻辑）
- 子类**可选实现** `_stream()`、`_agenerate()` 等

| 必须实现 | 可选实现 |
| :--- | :--- |
| `_generate` — 核心推理 | `_stream` — 流式输出 |
| `_llm_type` — 模型类型标识 | `_agenerate` — 异步推理 |
| | `_astream` — 异步流式 |

### 3. `RunnableSerializable` — 两大能力的融合

```python
class RunnableSerializable(Serializable, Runnable[Input, Output]):
```

融合了两个父类的能力：

| 来自 | 提供的能力 |
| :--- | :--- |
| **`Runnable`** | `invoke`、`stream`、`batch`、`\|` 管道组合 |
| **`Serializable`** | `to_json()`、`lc_secrets`（密钥脱敏）、`lc_id`（唯一标识） |

---

## ⚠️ 关于 Serializable 的注意事项

继承了 `Serializable` **不代表自动可序列化**，子类必须显式开启：

```python
# 默认 False → 不可序列化
@classmethod
def is_lc_serializable(cls) -> bool:
    return False  # 默认值

# 子类手动开启
@classmethod
def is_lc_serializable(cls) -> bool:
    return True   # 才能真正序列化
```

**设计原因**：防止包含 HTTP 客户端、API 密钥等敏感/不可序列化资源的对象被意外序列化。

---

## 🔗 相关源码

- `langchain_core/language_models/base.py` — `BaseLanguageModel` 定义
- `langchain_core/language_models/chat_models.py` — `BaseChatModel` 定义
- `langchain_core/load/serializable.py` — `Serializable` 定义
