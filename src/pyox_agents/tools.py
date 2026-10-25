"""Python tools: the `Tool` class and the `@tool` decorator."""

import asyncio
import inspect
import json
import types
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, Union, get_args, get_origin, get_type_hints, overload

JsonSchema = dict[str, Any]


class ToolLike(Protocol):
    """What the agent needs from a tool. `Tool` and Rust tools made with the
    `pyox-tools` crate both fit; `invoke` may be sync or async."""

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def parameters(self) -> JsonSchema: ...
    def invoke(self, arguments: dict[str, Any]) -> str | Awaitable[str]: ...


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool the model can call.

    `function` receives the arguments as keyword arguments. It can be sync or
    async, and may return a `str` or anything `json.dumps` accepts.
    Set `run_in_thread=True` for slow sync functions (for example a heavy Rust
    function from a compiled package) so they don't block the event loop.
    """

    name: str
    description: str
    parameters: JsonSchema
    function: Callable[..., Any]
    run_in_thread: bool = False

    async def invoke(self, arguments: dict[str, Any]) -> str:
        """Called by the Rust agent loop with the arguments chosen by the model."""
        if self.run_in_thread:
            result = await asyncio.to_thread(self.function, **arguments)
        else:
            result = self.function(**arguments)
            if inspect.isawaitable(result):
                result = await result
        return result if isinstance(result, str) else json.dumps(result)


@overload
def tool(function: Callable[..., Any], /) -> Tool: ...


@overload
def tool(
    *, name: str | None = None, description: str | None = None, run_in_thread: bool = False
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    function: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    run_in_thread: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into a `Tool`, building the JSON schema from its signature.

    Use `Annotated[str, "description"]` to describe a parameter.

        @tool
        def get_weather(city: Annotated[str, "City name"], unit: Literal["C", "F"] = "C") -> str:
            '''Get the current weather in a city.'''
    """

    def make_tool(function: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or getattr(function, "__name__", type(function).__name__),
            description=description or inspect.getdoc(function) or "",
            parameters=parameters_schema(function),
            function=function,
            run_in_thread=run_in_thread,
        )

    return make_tool if function is None else make_tool(function)


def parameters_schema(function: Callable[..., Any]) -> JsonSchema:
    """JSON schema of a function's parameters. Parameters without a default are required."""
    type_hints = get_type_hints(function, include_extras=True)
    properties: dict[str, JsonSchema] = {}
    required: list[str] = []
    for parameter in inspect.signature(function).parameters.values():
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
