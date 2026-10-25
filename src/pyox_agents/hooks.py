"""Hooks: functions the agent calls at fixed points of a run."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypedDict, TypeVar

from pyox_agents._core import RunContext, RunFinished
from pyox_agents.messages import AssistantMessage, Message, ToolCall


class ToolResult(TypedDict):
    """What a tool gave back, as the model will see it."""

    content: str
    is_error: bool


T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]


@dataclass(frozen=True, slots=True)
class Hooks:
    """Hooks made of plain functions (sync or async): `Agent(hooks=[Hooks(...)])`.

    Any object with some of these methods works too, and so do Rust hooks
    made with `pyox_tools::hooks`. Return `None` to leave things as they are,
    or a new value to replace them:

    - `before_run(ctx)`: once, when a run starts.
    - `before_model_request(ctx, messages)`: before each model request. Return
      other messages to send *for this request only*; the history stays as it is.
    - `after_model_response(ctx, message)`: after each model response, with the
      assistant message (its text was already streamed; its tools haven't run).
    - `before_tool_call(ctx, call)`: before each tool runs. Return a changed call
      (e.g. other `arguments`), or raise to block it: the model gets the error.
    - `after_tool_call(ctx, call, result)`: after each tool runs. Return a
      changed result. Raising also gives the model the error.
    - `after_run(ctx, result)`: once, when the run finishes successfully.

    Raising in the other hooks stops the run with that exception.

        def no_deletes(ctx: RunContext, call: ToolCall) -> None:
            if call["name"] == "delete_file":
                raise PermissionError("Deleting files is not allowed")

        agent = Agent(model="gpt-5-mini", tools=[...], hooks=[Hooks(before_tool_call=no_deletes)])
    """

    before_run: Callable[[RunContext], MaybeAwaitable[None]] | None = None
    before_model_request: Callable[[RunContext, list[Message]], MaybeAwaitable[list[Message] | None]] | None = None
    after_model_response: Callable[[RunContext, AssistantMessage], MaybeAwaitable[AssistantMessage | None]] | None = (
        None
    )
    before_tool_call: Callable[[RunContext, ToolCall], MaybeAwaitable[ToolCall | None]] | None = None
    after_tool_call: Callable[[RunContext, ToolCall, ToolResult], MaybeAwaitable[ToolResult | None]] | None = None
    after_run: Callable[[RunContext, RunFinished], MaybeAwaitable[None]] | None = None
