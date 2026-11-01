"""pyox's own event loop, for the async Python functions of sync runs (`agent.run_sync`).

It starts on first use, in a daemon thread, so sync runs with only sync tools never start it.
There is one per process: async clients made by one sync run (an `httpx.AsyncClient`, a
database pool, an MCP session) keep working in the next. A forked child starts its own.
"""

import asyncio
import concurrent.futures
import os
import threading
from contextlib import AbstractAsyncContextManager
from typing import Generic, TypeVar

T = TypeVar("T")

_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_pid: int | None = None


def get() -> asyncio.AbstractEventLoop:
    """The loop, started if needed."""
    global _loop, _pid
    with _lock:
        # In a forked child the loop's thread is gone: start another one.
        if _loop is None or _pid != os.getpid():
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="pyox-agents-loop", daemon=True).start()
            _pid = os.getpid()
        return _loop


def is_current() -> bool:
    """Whether this code runs on the loop (in an async tool of a sync run)."""
    try:
        return asyncio.get_running_loop() is _loop
    except RuntimeError:
        return False


class OpenContext(Generic[T]):
    """An async context manager (`async with context as value`) entered on the loop, and kept
    open until `close()`: `with McpTools(...)` for sync runs. One task enters and exits it, as
    anyio (which the `mcp` package uses) requires."""

    def __init__(self, context: AbstractAsyncContextManager[T]) -> None:
        self._entered: concurrent.futures.Future[T] = concurrent.futures.Future()
        self._closing = asyncio.Event()
        self._task = asyncio.run_coroutine_threadsafe(self._hold_open(context), get())
        self.value = self._entered.result()  # raises if entering failed

    async def _hold_open(self, context: AbstractAsyncContextManager[T]) -> None:
        try:
            async with context as value:
                self._entered.set_result(value)
                await self._closing.wait()
        except BaseException as error:
            if not self._entered.done():
                self._entered.set_exception(error)
            raise

    def close(self) -> None:
        get().call_soon_threadsafe(self._closing.set)
        self._task.result()
