# Runnables — LangChain 的核心协议

> `Runnable` 是 LangChain 的**绝对核心**，**且没有之一**。
> 无论是 Model、Tool、Prompt 还是 Parser，所有组件都实现了 `Runnable` 接口。
> LangChain 这样设计，我估计是有以下原因
>
> 1，为了不同的情况下，依旧能实现统一的调用方法，是高度抽象的设计。
>
> 2，这里吐槽一下，感觉是没有必要的东西，设计有点过于复杂

---

## 📦 `runnables/base.py` 核心类

```
Runnable (ABC, Generic[Input, Output])        ← 基类
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

LangChain 早期各组件调用方式不统一
`Runnable` 的出现将**所有组件统一为同一套方案**，解决了以下问题：

Runnable 的源码为以下，但是太长了，这里按下不表，后续再说

```python
class Runnable(ABC, Generic[Input, Output]):
    """A unit of work that can be invoked, batched, streamed, transformed and composed.

    Key Methods
    ===========

    - `invoke`/`ainvoke`: Transforms a single input into an output.
    - `batch`/`abatch`: Efficiently transforms multiple inputs into outputs.
    - `stream`/`astream`: Streams output from a single input as it's produced.
    - `astream_log`: Streams output and selected intermediate results from an
        input.
    name: str | None
    """The name of the `Runnable`. Used for debugging and tracing."""
```



1. **调用方法的统一** → 统一 `invoke`/`stream`/`batch`

2. **泛型推断**

   ```python
   @property
       def InputType(self) -> type[Input]:  # noqa: N802
           """Input type.
   
           The type of input this `Runnable` accepts specified as a type annotation.
   
           Raises:
               TypeError: If the input type cannot be inferred.
           """
           # First loop through all parent classes and if any of them is
           # a Pydantic model, we will pick up the generic parameterization
           # from that model via the __pydantic_generic_metadata__ attribute.
           for base in self.__class__.mro():
               if hasattr(base, "__pydantic_generic_metadata__"):
                   metadata = base.__pydantic_generic_metadata__
                   if (
                       "args" in metadata
                       and len(metadata["args"]) == _RUNNABLE_GENERIC_NUM_ARGS
                   ):
                       return cast("type[Input]", metadata["args"][0])
   
           # If we didn't find a Pydantic model in the parent classes,
           # then loop through __orig_bases__. This corresponds to
           # Runnables that are not pydantic models.
           for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
               type_args = get_args(cls)
               if type_args and len(type_args) == _RUNNABLE_GENERIC_NUM_ARGS:
                   return cast("type[Input]", type_args[0])
   
           msg = (
               f"Runnable {self.get_name()} doesn't have an inferable InputType. "
               "Override the InputType property to specify the input type."
           )
           raise TypeError(msg)
   
       @property
       def OutputType(self) -> type[Output]:  # noqa: N802
           """Output Type.
   
           The type of output this `Runnable` produces specified as a type annotation.
   
           Raises:
               TypeError: If the output type cannot be inferred.
           """
           # First loop through bases -- this will help generic
           # any pydantic models.
           for base in self.__class__.mro():
               if hasattr(base, "__pydantic_generic_metadata__"):
                   metadata = base.__pydantic_generic_metadata__
                   if (
                       "args" in metadata
                       and len(metadata["args"]) == _RUNNABLE_GENERIC_NUM_ARGS
                   ):
                       return cast("type[Output]", metadata["args"][1])
   
           for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
               type_args = get_args(cls)
               if type_args and len(type_args) == _RUNNABLE_GENERIC_NUM_ARGS:
                   return cast("type[Output]", type_args[1])
   
           msg = (
               f"Runnable {self.get_name()} doesn't have an inferable OutputType. "
               "Override the OutputType property to specify the output type."
           )
           raise TypeError(msg)
   ```

   

3. **组合式的执行** → `|` 其底层重写了 `__or__` 方法

   ```python
   def __or__(
           self,
           other: Runnable[Any, Other]
           | Callable[[Iterator[Any]], Iterator[Other]]
           | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
           | Callable[[Any], Other]
           | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
       ) -> RunnableSerializable[Input, Other]:
           """Runnable "or" operator.
   
           Compose this `Runnable` with another object to create a
           `RunnableSequence`.
   
           Args:
               other: Another `Runnable` or a `Runnable`-like object.
   
           Returns:
               A new `Runnable`.
           """
           return RunnableSequence(self, coerce_to_runnable(other))
   ```

   这使得封装出一个Sequence序列，将上一步的结果作为下一步组件的输出，当形成了 `langchain` 的组件之时例如以下例子。

   ```python
   chain = prompt | model    # 这里假设 prompt 为chatprompt之类的对象的时候， 由于 Runnable 重写了 __or__ 魔术方法
   chain = prompt.__or__(model) # 那么以上的动作就变成了这样子，使得其返回了 RunnableSequence 对象，当需要串行其他组件的时候，重复以上的操作即可
   ```

   **这便是`langchain` 最初串联组件的核心方式。**

   当然这里又出现了一个缺点，这就要回到 `Agent` 的定义上去了。

   什么是 `Agent` , 即 ***An LLM agent runs tools in a loop to achieve a goal***

   key point is ***the loop*** 但是其串行的方式意味着这无法进行自检和循环，这就不符合其定义

   因此 `langchain` 便推出了 `langgraph`  以及后续的大改版， 当然就这是其他模块要说的东西了。 

   ** **

4. **异步/流式重复写** → 基类提供默认实现

5. **类型不透明** → `input_schema`/`output_schema` 自动推断

---

## 🌟 第二部分：组合序列

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

## 🌟 第三部分：aliment of Agent 

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
