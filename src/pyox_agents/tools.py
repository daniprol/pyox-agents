"""Python tools: the `Tool` class and the `@tool` decorator, and the
exceptions a tool raises to choose what the model reads when it fails."""

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, get_type_hints, overload

from pyox_agents._core import RunContext
from pyox_agents.messages import ContentPart
from pyox_agents.schema import JsonSchema, from_json, json_schema, type_problems, validation_problems

ToolContent = str | list[str | ContentPart]
"""What the model reads from a tool: text, or a list of text, images and files."""


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
    def invoke(self, arguments: dict[str, Any]) -> "ToolContent | ToolReturn | Awaitable[ToolContent | ToolReturn]": ...


RequiresApproval = bool | Callable[[RunContext, dict[str, Any]], bool | Awaitable[bool]]
"""`True` to approve every call, or a function deciding per call from `ctx` and the arguments."""

OnToolError = Callable[[RunContext, Any, Exception], Any]
"""`on_error(ctx, call, error)`: what the model reads when a tool fails. Return `None` for the
default text or a `str`; raise (e.g. `StopRun`) to end the run. The function may be async.
Works like `Hooks.on_tool_error`, and is asked first."""


class ToolFailed(Exception):  # noqa: N818 (it reads as what happened, like `StopRun`)
    """Raise in a tool to tell the model it failed, with exactly this message
    (other exceptions are shown as "ValueError: message"). The run goes on, and
    the model decides what to do next.

        raise ToolFailed(f"No flights from {origin} on {date}. Try another date.")
    """


class InvalidArguments(ToolFailed):
    """The model called a tool with arguments that do not fit it (missing, of the
    wrong type, not valid JSON...). The model reads what is wrong, one line per
    problem, and decides whether to call the tool again. `errors` lists the problems."""

    def __init__(self, tool_name: str, errors: list[str]) -> None:
        lines = "\n".join(f"- {error}" for error in errors)
        super().__init__(f"Invalid arguments for {tool_name}:\n{lines}")
        self.tool_name = tool_name
        self.errors = errors

    def __reduce__(self) -> tuple[Any, ...]:  # for `pickle` and `copy`
        return (type(self), (self.tool_name, self.errors))


@dataclass(frozen=True, slots=True)
class ToolReturn:
    """A tool result with data for your app: the model reads `content` only.

        return ToolReturn(f"Booked {booking.id}", metadata={"receipt_url": booking.url})

    `content` may hold images and files too: `ToolReturn(["The page:", image(png_bytes, media_type="image/png")])`.
    `metadata` is kept in the history and on `ToolFinished.metadata`, never sent to the model.
    """

    content: Any  # text; a list of text, images and files; or anything `json.dumps` accepts
    metadata: Any = None


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool the model can call.

    `function` receives the arguments as keyword arguments. It can be sync or
    async, and may return a `str`, a `ToolReturn`, an `image(...)` or `file(...)`
    (or a list of them and text), or anything `json.dumps` accepts.
    With `takes_ctx=True` it also gets the run's `RunContext` first.
    Sync functions never block an event loop: async runs call them on worker
    threads, sync runs (`agent.run_sync`) on your own thread. Set
    `run_in_thread=True` to run them on worker threads in sync runs too, so
    several calls run in parallel.

    `requires_approval`: the call waits for a person's `Approve` before it runs
    (see `pyox_agents.human_input`). `frontend=True`: the tool has no Python
    code; the app runs it and sends back the result (`function` is never called).

    `on_error(ctx, call, error)`: chooses what the model reads when this tool
    fails, before the agent's `on_tool_error` hooks (see `OnToolError`).
    `timeout_seconds`: stop waiting for a call that takes longer; the model reads
    that it timed out. An async tool is cancelled; a sync one cannot be
    interrupted, and finishes on its own in the background.
    """

    name: str
    description: str
    parameters: JsonSchema
    function: Callable[..., Any]
    run_in_thread: bool = False
    takes_ctx: bool = False
    requires_approval: RequiresApproval = False
    frontend: bool = False
    on_error: OnToolError | None = None
    timeout_seconds: float | None = None
    argument_types: Mapping[str, Any] = field(default_factory=dict)
    """The type of each argument, set by `@tool`: the model's arguments are checked against
    them, and dataclasses and pydantic models built from JSON, before the call."""

    @staticmethod
    def frontend_tool(name: str, description: str, parameters: JsonSchema) -> "Tool":
        """A tool the app runs (for example in the browser), from its JSON schema:

        get_location = Tool.frontend_tool("get_location", "Get the user's location.", {"type": "object"})
        """
        return Tool(name, description, parameters, function=_runs_in_the_frontend, frontend=True)

    def invoke(self, arguments: dict[str, Any], ctx: RunContext | None = None) -> Any:
        """Called by the Rust agent loop with the arguments chosen by the model
        (and the context, when `takes_ctx` is set). A sync function runs right here
        (the agent loop picks the thread); an async one gives a coroutine to await."""
        positional = (ctx,) if self.takes_ctx else ()
        if self.argument_types:
            arguments = self._checked_arguments(arguments)
        result = self.function(*positional, **arguments)
        if inspect.isawaitable(result):
            return _tool_result_when_done(result)
        return _tool_result(result)

    def _checked_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """The model's arguments, checked against the parameter types, with dataclasses and
        pydantic models built from JSON. Raises `InvalidArguments` listing every problem."""
        problems = [f"{name}: missing" for name in self.parameters.get("required", []) if name not in arguments]
        checked = {}
        for name, value in arguments.items():
            if name not in self.argument_types:
                problems.append(f"{name}: not a parameter of this tool")
                continue
            annotation = self.argument_types[name]
            found = type_problems(annotation, value, name)
            if not found:
                try:
                    checked[name] = from_json(annotation, value)
                except Exception as error:  # a pydantic `ValidationError`, or a dataclass's own check
                    found = validation_problems(error, name)
            problems.extend(found)
        if problems:
            raise InvalidArguments(self.name, problems)
        return checked


async def _tool_result_when_done(result: Awaitable[Any]) -> ToolContent | ToolReturn:
    return _tool_result(await result)


def _tool_result(result: Any) -> ToolContent | ToolReturn:
    if isinstance(result, ToolReturn):
        return ToolReturn(_as_content(result.content), result.metadata)
    return _as_content(result)


_MEDIA_TYPES = ("image", "file")


def _as_content(result: Any) -> ToolContent:
    """Text stays text; images and files (alone or in a list with text) become content
    parts; anything else is sent as JSON text."""
    if isinstance(result, str):
        return result
    if _is_media(result):
        return [result]
    if isinstance(result, list) and any(_is_media(item) for item in result):
        return result
    return json.dumps(result)


def _is_media(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") in _MEDIA_TYPES and "url" in value


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
    on_error: OnToolError | None = None,
    timeout_seconds: float | None = None,
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
    on_error: OnToolError | None = None,
    timeout_seconds: float | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into a `Tool`, building the JSON schema from its signature.

    Parameters can be simple types, lists, `Literal`s, unions, dataclasses, `TypedDict`s
    or pydantic models (see `pyox_agents.schema`); dataclass and pydantic arguments
    arrive built from the model's JSON. Use `Annotated[str, "description"]` to describe a parameter. A first
    parameter typed `RunContext` is not shown to the model: it receives the
    run's context (`ctx.deps`, `ctx.conversation_id`, ...).

        @tool
        def get_weather(city: Annotated[str, "City name"], unit: Literal["C", "F"] = "C") -> str:
            '''Get the current weather in a city.'''

        @tool
        async def my_orders(ctx: RunContext[Database]) -> list[str]:
            '''List the user's orders.'''
            return await ctx.deps.orders(ctx.conversation_id)

    The model's arguments are checked against the parameter types before the call. When
    they do not fit, the model reads what is wrong and decides whether to try again.

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
            on_error=on_error,
            timeout_seconds=timeout_seconds,
            argument_types=argument_types(function),
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
    for parameter in _model_parameters(function):
        properties[parameter.name] = json_schema(type_hints.get(parameter.name, Any))
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
    # `json_schema` hoists nested `$defs` to the top; do the same across parameters.
    definitions = {name: value for schema in properties.values() for name, value in schema.pop("$defs", {}).items()}
    schema: JsonSchema = {"type": "object", "properties": properties, "required": required}
    return {**schema, "$defs": definitions} if definitions else schema


def argument_types(function: Callable[..., Any]) -> dict[str, Any]:
    """The type of each parameter the model fills in (`Any` when it has none). Empty for a
    function taking `**kwargs`, whose arguments cannot be checked."""
    type_hints = get_type_hints(function, include_extras=True)
    parameters = _model_parameters(function)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return {}
    return {parameter.name: type_hints.get(parameter.name, Any) for parameter in parameters}


def _model_parameters(function: Callable[..., Any]) -> list[inspect.Parameter]:
    """The parameters the model fills in: all but a first `RunContext` one."""
    parameters = list(inspect.signature(function).parameters.values())
    return parameters[1:] if takes_ctx(function) else parameters
