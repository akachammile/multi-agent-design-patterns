import asyncio
import threading
from contextvars import ContextVar, copy_context
from functools import partial


# Simulate request-scoped/tracing data.
trace_id: ContextVar[str] = ContextVar("trace_id", default="NO_TRACE")


def worker(label: str) -> str:
    """Function executed inside a thread-pool worker thread."""
    value = trace_id.get()
    print(f"[{label}] thread={threading.current_thread().name}, trace_id={value}")
    return value


async def main() -> None:
    loop = asyncio.get_running_loop()

    # Set context value in the main (event-loop) thread.
    trace_id.set("TRACE-123")
    print(f"[main] thread={threading.current_thread().name}, trace_id={trace_id.get()}")

    # Case 1: No context copy, worker usually won't see TRACE-123.
    result_no_copy = await loop.run_in_executor(None, worker, "no-copy")

    # Case 2: Copy context, then execute worker inside that copied context.
    # ctx = copy_context()
    # task = partial(ctx.run, worker, "with-copy")
    # result_with_copy = await loop.run_in_executor(None, task)

    print(f"result no-copy : {result_no_copy}")
    # print(f"result with-copy: {result_with_copy}")


if __name__ == "__main__":
    asyncio.run(main())
