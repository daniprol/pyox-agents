"""Lets the Rust agent loop cancel the async Python code it awaits."""

import asyncio
from collections.abc import Awaitable
from typing import Any


class Cancellable:
    """Runs an awaitable (a tool's coroutine, an async hook...) as a task that
    `cancel()` can cancel from any thread: the agent loop calls it when the run
    is stopped, cancelled or dropped, so the Python code does not run on alone."""

    def __init__(self, awaitable: Awaitable[Any]) -> None:
        self._awaitable = awaitable
        self._task: asyncio.Task[Any] | None = None
        self._cancelled = False

    async def run(self) -> Any:
        if self._cancelled:  # cancelled before it even started
            close = getattr(self._awaitable, "close", None)
            if close is not None:
                close()  # avoids the "coroutine was never awaited" warning
            raise asyncio.CancelledError
        self._task = asyncio.current_task()
        return await self._awaitable

    def cancel(self) -> None:
        self._cancelled = True
        if self._task is not None:
            self._task.get_loop().call_soon_threadsafe(self._task.cancel)
