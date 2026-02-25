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

## 🌟第二部分：一切都是Serializable之RunnableSerializable

~~~python
class RunnableSerializable(Serializable, Runnable[Input, Output]):
    """Runnable that can be serialized to JSON."""

    name: str | None = None
    """The name of the `Runnable`.

    Used for debugging and tracing.
    """

    model_config = ConfigDict(
        # Suppress warnings from pydantic protected namespaces
        # (e.g., `model_`)
        protected_namespaces=(),
    )

    @override
    def to_json(self) -> SerializedConstructor | SerializedNotImplemented:
        """Serialize the `Runnable` to JSON.

        Returns:
            A JSON-serializable representation of the `Runnable`.

        """
        dumped = super().to_json()
        with contextlib.suppress(Exception):
            dumped["name"] = self.get_name()
        return dumped

    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> RunnableSerializable[Input, Output]:
        """Configure particular `Runnable` fields at runtime.

        Args:
            **kwargs: A dictionary of `ConfigurableField` instances to configure.

        Raises:
            ValueError: If a configuration key is not found in the `Runnable`.

        Returns:
            A new `Runnable` with the fields configured.

        !!! example

            ```python
            from langchain_core.runnables import ConfigurableField
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(max_tokens=20).configurable_fields(
                max_tokens=ConfigurableField(
                    id="output_token_number",
                    name="Max tokens in the output",
                    description="The maximum number of tokens in the output",
                )
            )

            # max_tokens = 20
            print(
                "max_tokens_20: ", model.invoke("tell me something about chess").content
            )

            # max_tokens = 200
            print(
                "max_tokens_200: ",
                model.with_config(configurable={"output_token_number": 200})
                .invoke("tell me something about chess")
                .content,
            )
            ```
        """
        # Import locally to prevent circular import
        from langchain_core.runnables.configurable import (  # noqa: PLC0415
            RunnableConfigurableFields,
        )

        model_fields = type(self).model_fields
        for key in kwargs:
            if key not in model_fields:
                msg = (
                    f"Configuration key {key} not found in {self}: "
                    f"available keys are {model_fields.keys()}"
                )
                raise ValueError(msg)

        return RunnableConfigurableFields(default=self, fields=kwargs)

    def configurable_alternatives(
        self,
        which: ConfigurableField,
        *,
        default_key: str = "default",
        prefix_keys: bool = False,
        **kwargs: Runnable[Input, Output] | Callable[[], Runnable[Input, Output]],
    ) -> RunnableSerializable[Input, Output]:
        """Configure alternatives for `Runnable` objects that can be set at runtime.

        Args:
            which: The `ConfigurableField` instance that will be used to select the
                alternative.
            default_key: The default key to use if no alternative is selected.
            prefix_keys: Whether to prefix the keys with the `ConfigurableField` id.
            **kwargs: A dictionary of keys to `Runnable` instances or callables that
                return `Runnable` instances.

        Returns:
            A new `Runnable` with the alternatives configured.

        !!! example

            ```python
            from langchain_anthropic import ChatAnthropic
            from langchain_core.runnables.utils import ConfigurableField
            from langchain_openai import ChatOpenAI

            model = ChatAnthropic(
                model_name="claude-sonnet-4-5-20250929"
            ).configurable_alternatives(
                ConfigurableField(id="llm"),
                default_key="anthropic",
                openai=ChatOpenAI(),
            )

            # uses the default model ChatAnthropic
            print(model.invoke("which organization created you?").content)

            # uses ChatOpenAI
            print(
                model.with_config(configurable={"llm": "openai"})
                .invoke("which organization created you?")
                .content
            )
            ```
        """
        # Import locally to prevent circular import
        from langchain_core.runnables.configurable import (  # noqa: PLC0415
            RunnableConfigurableAlternatives,
        )

        return RunnableConfigurableAlternatives(
            which=which,
            default=self,
            alternatives=kwargs,
            default_key=default_key,
            prefix_keys=prefix_keys,
        )

~~~

其承载的核心功能就是Serialize所有可Serialize的Runnable的对象，langchain重写了Serializable，填充了关于lc的一堆属性，如下

```python
    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example, `{"openai_api_key": "OPENAI_API_KEY"}`
        """
        return {}

    @property
    def lc_attributes(self) -> dict:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.

        Default is an empty dictionary.
        """
        return {}

    @classmethod
    def lc_id(cls) -> list[str]:
        """Return a unique identifier for this class for serialization purposes.

        The unique identifier is a list of strings that describes the path
        to the object.

        For example, for the class `langchain.llms.openai.OpenAI`, the id is
        `["langchain", "llms", "openai", "OpenAI"]`.
```

等等方法，在langchain中万物皆对象，废话其实，对象就有独一无二的属性。

## 🌟 第三部分：组合序列

### 一，`RunnableSequence` — 串行链

```python
chain = prompt | model | parser
# 内部：RunnableSequence(first=prompt, middle=[model], last=parser)
# 执行：prompt 的输出 → model 的输入 → parser 的输入
```

`|` 操作符就是 `__or__` 重载，返回一个 `RunnableSequence` 对象。

因此，当Runnable对象使用 `__or__` 方法的时候，Runnable对象自己就变成了  `RunnableSequence`

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

这里`coerce_to_runnable` 会把类Runnable的所有类转成Runnable, 也是为了统一

### 二，`RunnableParallel` — 并行链

官方在注释中写明了

***  ***

***RunnableParallel is one of the two main composition primitives***

嘛意思呢？

大白话就是，RunnalbeParallel 是非常重要组合件之一，另外一个是嘛呢，就是上面的RunnableSequenece 

在这里说下这两种方式有何不同

**我们之前提到过了，RunnableSequence 是 Runnable 调用 or 方法后返回的结果，那么 Sequence 究竟产生了一个什么结果呢？**看下他的init方法

```python
first: Runnable[Input, Any]
    """The first `Runnable` in the sequence."""
    middle: list[Runnable[Any, Any]] = Field(default_factory=list)
    """The middle `Runnable` in the sequence."""
    last: Runnable[Any, Output]
    """The last `Runnable` in the sequence."""

    def __init__(
        self,
        *steps: RunnableLike,
        name: str | None = None,
        first: Runnable[Any, Any] | None = None,
        middle: list[Runnable[Any, Any]] | None = None,
        last: Runnable[Any, Any] | None = None,
    ) -> None:
        """Create a new `RunnableSequence`.

        Args:
            steps: The steps to include in the sequence.
            name: The name of the `Runnable`.
            first: The first `Runnable` in the sequence.
            middle: The middle `Runnable` objects in the sequence.
            last: The last `Runnable` in the sequence.

        Raises:
            ValueError: If the sequence has less than 2 steps.
        """
        steps_flat: list[Runnable] = []
        if not steps and first is not None and last is not None:
            steps_flat = [first] + (middle or []) + [last]
        for step in steps:
            if isinstance(step, RunnableSequence):
                steps_flat.extend(step.steps)
            else:
                steps_flat.append(coerce_to_runnable(step))
        if len(steps_flat) < _RUNNABLE_SEQUENCE_MIN_STEPS:
            msg = (
                f"RunnableSequence must have at least {_RUNNABLE_SEQUENCE_MIN_STEPS} "
                f"steps, got {len(steps_flat)}"
            )
            raise ValueError(msg)
        super().__init__(
            first=steps_flat[0],
            middle=list(steps_flat[1:-1]),
            last=steps_flat[-1],
            name=name,
        )
```

这里 RunnableSequence 方法，定义了三个参数，first、middle、last 首尾参数都是一个 Runnable 对象，中间是一个 list 的 Runnable 对象。

再结合Sequence这个方法名，显而易见，这是一个顺序的链条，下面再看其是如何拼接的

```python
        steps_flat: list[Runnable] = []
        if not steps and first is not None and last is not None:
            steps_flat = [first] + (middle or []) + [last]
        for step in steps:
            if isinstance(step, RunnableSequence):
                steps_flat.extend(step.steps)
            else:
                steps_flat.append(coerce_to_runnable(step))
        if len(steps_flat) < _RUNNABLE_SEQUENCE_MIN_STEPS:
            msg = (
                f"RunnableSequence must have at least {_RUNNABLE_SEQUENCE_MIN_STEPS} "
                f"steps, got {len(steps_flat)}"
            )
            raise ValueError(msg)
        super().__init__(
            first=steps_flat[0],
            middle=list(steps_flat[1:-1]),
            last=steps_flat[-1],
            name=name,
        )
```



其他的不赘述，要注意一点，就是当，step，即中间的一堆存在时，直接会用extend方法重构下，最后调用 pydantic 完成整体的验证

相比之下，RunnableParallel 的构建方式就 复杂一点，其规定了形成方式是 key-value 的形式，因此其初始化的时候

形成了的形式，其中 `coerce_to_runnable` 是一个强制转换的方法

```python
steps__={key: coerce_to_runnable(r) for key, r in merged.items()}
```



```python
class RunnableParallel(RunnableSerializable[Input, dict[str, Any]]):
    """Runnable that runs a mapping of `Runnable`s in parallel.

    Returns a mapping of their outputs.

    `RunnableParallel` is one of the two main composition primitives,
    alongside `RunnableSequence`. It invokes `Runnable`s concurrently, providing the
    same input to each.

    A `RunnableParallel` can be instantiated directly or by using a dict literal
    within a sequence.

    Here is a simple example that uses functions to illustrate the use of
    `RunnableParallel`:

        ```python
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

        sequence = runnable_1 | {  # this dict is coerced to a RunnableParallel
            "mul_two": runnable_2,
            "mul_three": runnable_3,
        }
        # Or equivalently:
        # sequence = runnable_1 | RunnableParallel(
        #     {"mul_two": runnable_2, "mul_three": runnable_3}
        # )
        # Also equivalently:
        # sequence = runnable_1 | RunnableParallel(
        #     mul_two=runnable_2,
        #     mul_three=runnable_3,
        # )

        sequence.invoke(1)
        await sequence.ainvoke(1)

        sequence.batch([1, 2, 3])
        await sequence.abatch([1, 2, 3])
```

~~~python
`RunnableParallel` makes it easy to run `Runnable`s in parallel. In the below
example, we simultaneously stream output from two different `Runnable` objects:

    ```python
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI()
    joke_chain = (
        ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
    )
    poem_chain = (
        ChatPromptTemplate.from_template("write a 2-line poem about {topic}")
        | model
    )

    runnable = RunnableParallel(joke=joke_chain, poem=poem_chain)

    # Display stream
    output = {key: "" for key, _ in runnable.output_schema()}
    for chunk in runnable.stream({"topic": "bear"}):
        for key in chunk:
            output[key] = output[key] + chunk[key].content
        print(output)  # noqa: T201
    ```


steps__: Mapping[str, Runnable[Input, Any]]

def __init__(
    self,
    steps__: Mapping[
        str,
        Runnable[Input, Any]
        | Callable[[Input], Any]
        | Mapping[str, Runnable[Input, Any] | Callable[[Input], Any]],
    ]
    | None = None,
    **kwargs: Runnable[Input, Any]
    | Callable[[Input], Any]
    | Mapping[str, Runnable[Input, Any] | Callable[[Input], Any]],
) -> None:
    """Create a `RunnableParallel`.

    Args:
        steps__: The steps to include.
        **kwargs: Additional steps to include.

    """
    merged = {**steps__} if steps__ is not None else {}
    merged.update(kwargs)
    super().__init__(
        steps__={key: coerce_to_runnable(r) for key, r in merged.items()}
   )
~~~
`RunnableParallel` 的初始化运行是在invoke阶段完成的(也是废话)，其实都是这个阶段运行的，只不过是这个特殊一点，

值得注意的是，其底 RunnableParallel 底层用了一个 继承了 `ThreadPoolExecutor` 的 · `ContextThreadPoolExecutor` 

其同步方法用到的是线程池的方案，同时保留了上下文的信息，具体是怎么做到了呢？





### 三，`RunnableGenerator` — 生成器包装器

该部分不太重要，主要是包装一个底层的迭代器了，其他的无他

```python
def stream_words(input):
    for word in input.split():
        yield word

streamer = RunnableGenerator(stream_words)  # 支持流式
```

---

### 四，`RunnableEach` — **Each 运行单元**

该部分底层用到的是asynio的gather方法，然后遍历`config`

```python
class RunnableEachBase(RunnableSerializable[list[Input], list[Output]]):
    """RunnableEachBase class.

    `Runnable` that calls another `Runnable` for each element of the input sequence.

    Use only if creating a new `RunnableEach` subclass with different `__init__`
    args.

    See documentation for `RunnableEach` for more details.

    """

    bound: Runnable[Input, Output]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @override
    def InputType(self) -> Any:
        return list[self.bound.InputType]  # type: ignore[name-defined]

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        return create_model_v2(
            self.get_name("Input"),
            root=(
                list[self.bound.get_input_schema(config)],  # type: ignore[misc]
                None,
            ),
            # create model needs access to appropriate type annotations to be
            # able to construct the Pydantic model.
            # When we create the model, we pass information about the namespace
            # where the model is being created, so the type annotations can
            # be resolved correctly as well.
            # self.__class__.__module__ handles the case when the Runnable is
            # being sub-classed in a different module.
            module_name=self.__class__.__module__,
        )

    @property
    @override
    def OutputType(self) -> type[list[Output]]:
        return list[self.bound.OutputType]  # type: ignore[name-defined]

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        schema = self.bound.get_output_schema(config)
        return create_model_v2(
            self.get_name("Output"),
            root=list[schema],  # type: ignore[valid-type]
            # create model needs access to appropriate type annotations to be
            # able to construct the Pydantic model.
            # When we create the model, we pass information about the namespace
            # where the model is being created, so the type annotations can
            # be resolved correctly as well.
            # self.__class__.__module__ handles the case when the Runnable is
            # being sub-classed in a different module.
            module_name=self.__class__.__module__,
        )

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self.bound.config_specs

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        return self.bound.get_graph(config)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    def _invoke(
        self,
        inputs: list[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> list[Output]:
        configs = [
            patch_config(config, callbacks=run_manager.get_child()) for _ in inputs
        ]
        return self.bound.batch(inputs, configs, **kwargs)

    @override
    def invoke(
        self, input: list[Input], config: RunnableConfig | None = None, **kwargs: Any
    ) -> list[Output]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        inputs: list[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> list[Output]:
        configs = [
            patch_config(config, callbacks=run_manager.get_child()) for _ in inputs
        ]
        return await self.bound.abatch(inputs, configs, **kwargs)

    @override
    async def ainvoke(
        self, input: list[Input], config: RunnableConfig | None = None, **kwargs: Any
    ) -> list[Output]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)
```



## 🌟 第四部分：aliment of Runnable

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



---

## 🔗 相关源码

- `langchain_core/runnables/base.py` — `Runnable` 及所有组合原语的定义
- `langchain_core/runnables/config.py` — `RunnableConfig` 运行时配置
