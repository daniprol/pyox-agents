"""Python tools: the `Tool` class and the `@tool` decorator."""

import asyncio
import inspect
import json
import types
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, Union, get_args, get_origin, get_type_hints, overload

from pyox_agents._core import RunContext

JsonSchema = dict[str, Any]


class ToolLike(Protocol):
    """What the agent needs from a tool. `Tool` and Rust tools made with the
    `pyox-tools` crate both fit; `invoke` may be sync or async. A tool with a
    `takes_ctx = True` attribute is called as `invoke(arguments, ctx)`.
    Optional attributes: `requires_approval` and `frontend` (see `Tool`)."""

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def parameters(self) -> JsonSchema: ...
    def invoke(self, arguments: dict[str, Any]) -> "str | ToolReturn | Awaitable[str | ToolReturn]": ...


RequiresApproval = bool | Callable[[RunContext, dict[str, Any]], bool | Awaitable[bool]]
"""`True` to approve every call, or a function deciding per call from `ctx` and the arguments."""


@dataclass(frozen=True, slots=True)
class ToolReturn:
    """A tool result with data for your app: the model reads `content` only.

        return ToolReturn(f"Booked {booking.id}", metadata={"receipt_url": booking.url})

    `metadata` is kept in the history and on `ToolFinished.metadata`, never sent to the model.
    """

    content: Any  # text, or anything `json.dumps` accepts
    metadata: Any = None


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool the model can call.

    `function` receives the arguments as keyword arguments. It can be sync or
    async, and may return a `str`, a `ToolReturn`, or anything `json.dumps` accepts.
    With `takes_ctx=True` it also gets the run's `RunContext` first.
    Set `run_in_thread=True` for slow sync functions (for example a heavy Rust
    function from a compiled package) so they don't block the event loop.

    `requires_approval`: the call waits for a person's `Approve` before it runs
    (see `pyox_agents.human_input`). `frontend=True`: the tool has no Python
    code; the app runs it and sends back the result (`function` is never called).
    """

    name: str
    description: str
    parameters: JsonSchema
    function: Callable[..., Any]
    run_in_thread: bool = False
    takes_ctx: bool = False
    requires_approval: RequiresApproval = False
    frontend: bool = False

    @staticmethod
    def frontend_tool(name: str, description: str, parameters: JsonSchema) -> "Tool":
        """A tool the app runs (for example in the browser), from its JSON schema:

        get_location = Tool.frontend_tool("get_location", "Get the user's location.", {"type": "object"})
        """
        return Tool(name, description, parameters, function=_runs_in_the_frontend, frontend=True)

    async def invoke(self, arguments: dict[str, Any], ctx: RunContext | None = None) -> str | ToolReturn:
        """Called by the Rust agent loop with the arguments chosen by the model
        (and the context, when `takes_ctx` is set)."""
        positional = (ctx,) if self.takes_ctx else ()
        if self.run_in_thread:
            result = await asyncio.to_thread(self.function, *positional, **arguments)
        else:
            result = self.function(*positional, **arguments)
            if inspect.isawaitable(result):
                result = await result
        if isinstance(result, ToolReturn):
            return ToolReturn(_as_text(result.content), result.metadata)
        return _as_text(result)


def _as_text(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result)


def _runs_in_the_frontend(**arguments: Any) -> Any:
    raise RuntimeError("This is a frontend tool: the app runs it, not Python")


@overload
def tool(function: Callable[..., Any], /) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    run_in_thread: bool = False,
    requires_approval: RequiresApproval = False,
    frontend: bool = False,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    function: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    run_in_thread: bool = False,
    requires_approval: RequiresApproval = False,
    frontend: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into a `Tool`, building the JSON schema from its signature.

    Use `Annotated[str, "description"]` to describe a parameter. A first
    parameter typed `RunContext` is not shown to the model: it receives the
    run's context (`ctx.deps`, `ctx.conversation_id`, ...).

        @tool
        def get_weather(city: Annotated[str, "City name"], unit: Literal["C", "F"] = "C") -> str:
            '''Get the current weather in a city.'''

        @tool
        async def my_orders(ctx: RunContext[Database]) -> list[str]:
            '''List the user's orders.'''
            return await ctx.deps.orders(ctx.conversation_id)

    `requires_approval=True` makes each call wait for a person's approval.
    `frontend=True` declares a tool the app runs: only the signature and the
    docstring are used, the body never runs.

        @tool(frontend=True)
        def get_location(precision: Literal["city", "exact"] = "city") -> str:
            '''Get the user's current location from their device.'''
    """

    def make_tool(function: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or getattr(function, "__name__", type(function).__name__),
            description=description or inspect.getdoc(function) or "",
            parameters=parameters_schema(function),
            function=function,
            run_in_thread=run_in_thread,
            takes_ctx=takes_ctx(function),
            requires_approval=requires_approval,
            frontend=frontend,
        )

    return make_tool if function is None else make_tool(function)


def takes_ctx(function: Callable[..., Any]) -> bool:
    """Whether the first parameter of `function` is typed `RunContext`."""
    parameters = list(inspect.signature(function).parameters.values())
    # `RunContext[Deps]` is `RunContext` itself at runtime.
    return bool(parameters) and get_type_hints(function).get(parameters[0].name) is RunContext


def parameters_schema(function: Callable[..., Any]) -> JsonSchema:
    """JSON schema of a function's parameters (without the `RunContext` one).
    Parameters without a default are required."""
    type_hints = get_type_hints(function, include_extras=True)
    properties: dict[str, JsonSchema] = {}
    required: list[str] = []
    parameters = list(inspect.signature(function).parameters.values())
    for parameter in parameters[1:] if takes_ctx(function) else parameters:
        properties[parameter.name] = type_schema(type_hints.get(parameter.name, Any))
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
    return {"type": "object", "properties": properties, "required": required}


_SIMPLE_TYPES: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def type_schema(annotation: Any) -> JsonSchema:
    """JSON schema for one Python type annotation."""
    origin, args = get_origin(annotation), get_args(annotation)

    if annotation in _SIMPLE_TYPES:
        return {"type": _SIMPLE_TYPES[annotation]}
    if origin is Annotated:
        schema = type_schema(args[0])
        descriptions = [extra for extra in args[1:] if isinstance(extra, str)]
        return {**schema, "description": descriptions[0]} if descriptions else schema
    if origin is Literal:
        return {"enum": list(args)}
    if origin in (Union, types.UnionType):
        options = [type_schema(arg) for arg in args if arg is not type(None)]
        return options[0] if len(options) == 1 else {"anyOf": options}
    if annotation is list or origin is list:
        return {"type": "array", "items": type_schema(args[0])} if args else {"type": "array"}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    return {}  # Any or unknown: accept any JSON value
