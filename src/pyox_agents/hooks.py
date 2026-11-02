"""Hooks: functions the agent calls at fixed points of a run."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict, TypeVar

from pyox_agents._core import RunContext, RunFinished
from pyox_agents.messages import AssistantMessage, ContentPart, Message, ToolCall


class ModelRequest(TypedDict):
    """What the next model request sends. `before_model_request` hooks may change it, for that request only."""

    instructions: str  # the system prompt (`Agent(instructions=...)`)
    messages: list[Message]  # what to send; the history itself never changes
    tools: list[str]  # names of the tools to offer, in order: some of the run's tools
    model_parameters: dict[str, Any]  # merged into the request body: `tool_choice`, `reasoning`, ...


class ToolResult(TypedDict):
    """What a tool gave back, as the model will see it."""

    content: str | list[ContentPart]  # parts when the result has images or files
    is_error: bool
    metadata: NotRequired[Any]  # from `ToolReturn(metadata=...)`; never sent to the model


T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]


@dataclass(frozen=True, slots=True)
class Hooks:
    """Hooks made of plain functions (sync or async): `Agent(hooks=[Hooks(...)])`.

    Any object with some of these methods works too, and so do Rust hooks
    made with `pyox_tools::hooks`. Return `None` to leave things as they are,
    or a new value to replace them:

    - `before_run(ctx)`: once, when a run starts.
    - `before_model_request(ctx, request)`: before each model request, with the
      `ModelRequest` (instructions, messages, tool names, model parameters).
      Return a changed request to send *for this request only*; the history
      and the agent stay as they are.
    - `after_model_response(ctx, message)`: after each model response, with the
      assistant message (its text was already streamed; its tools haven't run).
    - `before_tool_call(ctx, call)`: before each tool runs. Return a changed call
      (e.g. other `arguments`), or raise to block it: the model gets the error.
    - `after_tool_call(ctx, call, result)`: after each tool runs. Return a
      changed result. Raising also gives the model the error.
    - `on_tool_error(ctx, call, error)`: when a tool fails, with the exception
      itself (also for bad arguments, an unknown tool or a timeout). Return a
      `str` for the model to read instead of the default, or `None` to keep it.
      Raise (`StopRun`, say) to end the run. A tool's own `@tool(on_error=...)`
      is asked first.
    - `after_run(ctx, result)`: once, when the run finishes successfully.

    Raising in the other hooks ends the run with a `HookError` (your exception
    is its `__cause__`); raise `StopRun` to end it with your own message.

        def no_deletes(ctx: RunContext, call: ToolCall) -> None:
            if call["name"] == "delete_file":
                raise PermissionError("Deleting files is not allowed")

        agent = Agent(model="gpt-5-mini", tools=[...], hooks=[Hooks(before_tool_call=no_deletes)])
    """

    before_run: Callable[[RunContext], MaybeAwaitable[None]] | None = None
    before_model_request: Callable[[RunContext, ModelRequest], MaybeAwaitable[ModelRequest | None]] | None = None
    after_model_response: Callable[[RunContext, AssistantMessage], MaybeAwaitable[AssistantMessage | None]] | None = (
        None
    )
    before_tool_call: Callable[[RunContext, ToolCall], MaybeAwaitable[ToolCall | None]] | None = None
    after_tool_call: Callable[[RunContext, ToolCall, ToolResult], MaybeAwaitable[ToolResult | None]] | None = None
    on_tool_error: Callable[[RunContext, ToolCall, Exception], MaybeAwaitable[str | ToolResult | None]] | None = None
    after_run: Callable[[RunContext, RunFinished], MaybeAwaitable[None]] | None = None
