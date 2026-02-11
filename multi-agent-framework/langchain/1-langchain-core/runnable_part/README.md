# Runnables — LangChain 的核心协议

> `Runnable` 是 LangChain 的绝对核心，且没有之一。
> 无论是 Model、Tool、Prompt 还是 Parser，所有组件都实现了 `Runnable` 接口。
> LangChain 这样设计，我估计是为了不同的情况下，依旧能实现统一的调用方法，是高度抽象的设计。

---

## 📦 `runnables/base.py` 核心类一览

```
Runnable (ABC, Generic[Input, Output])        ← 万物基类（6258 行）
    │
    ├── RunnableSerializable                  ← 可序列化的 Runnable
    │
    ├── RunnableSequence                      ← 串行链（A | B | C）
    │
    ├── RunnableParallel                      ← 并行链（{"a": A, "b": B}）
    │
    ├── RunnableGenerator                     ← 生成器函数包装器
    │
    └── RunnableLambda                        ← 普通函数包装器
```



---

## 🌟 第一部分：Runnable 接口定义

### 核心方法（4 类）

| 分类 | 方法 | 说明 |
| :--- | :--- | :--- |
| **执行** | `invoke` / `ainvoke` | 单次调用（同步/异步） |
| | `stream` / `astream` | 流式输出 |
| | `batch` / `abatch` | 批量并发 |
| | `transform` / `atransform` | 流式输入 → 流式输出 |
| **组合** | `__or__` (`\|`) | 串行组合：`A \| B \| C` → `RunnableSequence` |
| | `pipe()` | 同上，方法调用版 |
| | `pick()` | 从 dict 输出中选 key |
| | `assign()` | 给 dict 输出添加新 key |
| **装饰** | `bind()` | 绑定默认参数（Agent 绑定工具的基础） |
| | `with_config()` | 绑定运行时配置 |
| | `with_retry()` | 失败自动重试 |
| | `with_fallbacks()` | 失败切换备用方案 |
| | `with_listeners()` | 添加生命周期钩子 |
| **内省** | `input_schema` / `output_schema` | 获取输入/输出的 Pydantic Schema |
| | `get_graph()` | 获取图结构（可视化用） |

### 设计初衷

LangChain 早期各组件调用方式不统一，
`Runnable` 的出现将**所有组件统一为同一套接口**，解决了：

1. **调用碎片化** → 统一 `invoke`/`stream`/`batch`
2. **组合很麻烦** → `|` 管道符一行搞定
3. **异步/流式重复写** → 基类提供默认实现
4. **类型不透明** → `input_schema`/`output_schema` 自动推断

---

## 🌟 第二部分：组合原语

### `RunnableSequence` — 串行链（最常用）

```python
chain = prompt | model | parser
# 内部：RunnableSequence(first=prompt, middle=[model], last=parser)
# 执行：prompt 的输出 → model 的输入 → parser 的输入
```

`|` 操作符就是 `__or__` 重载，返回一个 `RunnableSequence` 对象。

### `RunnableParallel` — 并行链

```python
chain = prompt | {"answer": model, "source": retriever}
# 字典字面量自动变成 RunnableParallel
# 同一个输入同时发给 model 和 retriever，结果合成一个 dict
```

### `RunnableLambda` — 普通函数包装器

```python
add_one = RunnableLambda(lambda x: x + 1)
chain = add_one | model  # 普通函数也能参与链式调用
```

### `RunnableGenerator` — 生成器包装器

```python
def stream_words(input):
    for word in input.split():
        yield word

streamer = RunnableGenerator(stream_words)  # 支持流式
```

---

## 🌟 第三部分：Agent 面试必知

### `bind()` — Agent 绑定工具的基础

```python
model_with_tools = model.bind_tools(tools)
# 底层就是 bind()，将工具 schema 作为默认参数绑定到模型上
```

### `with_retry()` + `with_fallbacks()` — 容错机制

```python
safe_model = model.with_retry(stop_after_attempt=3)
safe_model = gpt4.with_fallbacks([gpt35, local_model])
```

### LCEL 的现状（2025）

LCEL（`A | B | C` 语法）没有被废弃，但已**从台前退到台后**：

| 层级 | 用什么 | 说明 |
| :--- | :--- | :--- |
| **Agent 编排层** | **LangGraph** | 状态机 + 图（支持循环和分支） |
| **单步执行层** | **LCEL** | 在 LangGraph 节点内部 `prompt \| model \| parser` |

LCEL 适合线性链，但 Agent 需要循环（思考→行动→观察→再思考），这需要 LangGraph。

---

## 🔗 相关源码

- `langchain_core/runnables/base.py` — `Runnable` 及所有组合原语的定义
- `langchain_core/runnables/config.py` — `RunnableConfig` 运行时配置
