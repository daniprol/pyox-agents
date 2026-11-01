"""MCP: give an agent the tools of any MCP server.

    from pyox_agents import Agent
    from pyox_agents.mcp import McpTools

    async with McpTools.http("https://mcp.deepwiki.com/mcp") as deepwiki:
        agent = Agent(model="gpt-5-mini", tools=deepwiki.tools)
        print((await agent.run("How does pydantic-ai run MCP servers?")).text)

`McpTools` connects to one server (a URL, or a command it starts and talks to over
stdin/stdout), lists its tools once, and gives them as `server.tools`: ordinary agent tools,
so approvals, hooks, subagents and web streams work on them as on any other tool.

When the server asks the user something in the middle of a tool call (MCP "elicitation"), the
question becomes a `ctx.ask(...)`: a live `on_request` handler answers it, or the run stops with
the question in `RunFinished.pending` and goes on when a later run brings the answer (see
`pyox_agents.human_input`). The person sees `InputRequested.data`:

    {"type": "mcp_elicitation", "server": "rooms", "mode": "form", "message": "Which floor?", "schema": {...}}

`Answer({"floor": 3})` accepts with that content; `Deny()` declines and ends the tool call. In
`"url"` mode, `data` has the `"url"` to open instead of a schema, and `Answer()` means done.

Uses the official `mcp` package, which speaks every MCP version (2024-11-05 to 2026-07-28):
`pip install "pyox-agents[mcp]"`. Anything it can do that this module does not wrap is one
step away: `server.client` is its `mcp.Client`, and extra keyword arguments of `McpTools(...)`
go to it. To change how tools, results or questions are converted, subclass `McpTools` and
override `make_tool`, `convert_result` or `elicit`.
"""

import itertools
import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import AsyncExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, ClassVar, Literal, Self

try:
    import httpx2
    import mcp
    import mcp_types
    from mcp.client.session import ClientRequestContext
    from mcp.client.streamable_http import streamable_http_client
except ImportError as error:  # pragma: no cover (the tests run with the extra)
    raise ImportError('MCP support needs the mcp extra: pip install "pyox-agents[mcp]"') from error

from pyox_agents._core import RunContext, StopToolCall
from pyox_agents.messages import ContentPart, Message
from pyox_agents.schema import JsonSchema
from pyox_agents.tools import RequiresApproval, ToolContent, ToolReturn

ApprovalPolicy = bool | Literal["unless_read_only"] | Collection[str] | Callable[[mcp_types.Tool], RequiresApproval]
"""Which tools of the server need a person's approval before each call:

- `False` (default) or `True`: none, or all of them.
- `"unless_read_only"`: every tool the server did not mark as read-only (`readOnlyHint`).
  The marks come from the server, so they are a convenience, not a security boundary.
- a set of tool names, as the server names them (without `prefix`).
- a function getting each `mcp_types.Tool` and returning what `@tool(requires_approval=...)` takes.
"""


class McpError(Exception):
    """An MCP server could not be used: not connected, or it failed to answer."""


class McpToolError(Exception):
    """An MCP tool reported an error (`isError`). The model reads its message, as for any tool error."""


@dataclass(frozen=True, slots=True)
class McpTool:
    """One tool of an MCP server, ready for an agent. `McpTools` makes them; see `make_tool`."""

    name: str  # as the model sees it: with the prefix, and only characters OpenAI accepts
    description: str
    parameters: JsonSchema
    requires_approval: RequiresApproval
    info: mcp_types.Tool  # the tool as the server listed it: its own name, annotations, output schema...
    server: "McpTools" = field(repr=False)
    takes_ctx: ClassVar[bool] = True  # `invoke` gets the run's context, for questions from the server

    async def invoke(self, arguments: dict[str, Any], ctx: RunContext | None = None) -> ToolContent | ToolReturn:
        return await self.server.call_tool(self.info.name, arguments, ctx)


@dataclass(eq=False)
class _ToolCall:
    """One tool call in progress, so the server's questions reach the run that made it."""

    ctx: RunContext
    stop: StopToolCall | None = None  # set when a question must wait for a person (or was denied)


# The number of the tool call in progress (a key of `McpTools._calls`). Not the call itself: the
# `mcp` package may start tasks that keep a copy of this context after the call, and holding the
# run's context there would keep the run from ending.
_current_call: ContextVar[int | None] = ContextVar("pyox_mcp_tool_call", default=None)
_call_numbers = itertools.count()


@dataclass(frozen=True, slots=True)
class _Http:
    """A Streamable HTTP server that needs headers or auth: its HTTP client is made on connect."""

    url: str
    headers: Mapping[str, str] | None
    auth: "httpx2.Auth | None"


class McpTools:
    """The tools of one MCP server. Use it with `async with`, which connects and lists the tools:

        async with McpTools.stdio("uvx", ["mcp-server-fetch"]) as fetch:
            agent = Agent(model="gpt-5-mini", tools=fetch.tools)

    The connection stays open until the block ends; the agent's calls fail after that (the model
    reads the error). Make it with `McpTools.http(...)` or `McpTools.stdio(...)`, or give
    `McpTools(server)` anything `mcp.Client` connects to: a URL, `mcp.StdioServerParameters`, a
    transport (for example with OAuth), or an `MCPServer` object in the same process (handy in tests).

    - `prefix`: put before each tool name (`prefix="gh"` makes `gh_search`), to avoid clashes
      between servers.
    - `include` / `exclude`: keep only, or leave out, these tools (names as the server gives them).
      Big servers have dozens of tools, and each one costs tokens.
    - `requires_approval`: which tools wait for a person's approval (see `ApprovalPolicy`).
    - Any other keyword goes to `mcp.Client`: `read_timeout_seconds=`, `mode="legacy"`,
      `client_info=`, ... An `elicitation_callback=` there replaces the questions-as-`ctx.ask` behaviour.
    """

    def __init__(
        self,
        server: Any,
        *,
        prefix: str | None = None,
        include: Collection[str] | None = None,
        exclude: Collection[str] = (),
        requires_approval: ApprovalPolicy = False,
        **client_options: Any,
    ) -> None:
        self._server = server
        self._prefix = prefix
        self._include = include
        self._exclude = exclude
        self._requires_approval = requires_approval
        self._client_options = client_options
        self._client: mcp.Client | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: tuple[McpTool, ...] | None = None
        self._calls: dict[int, _ToolCall] = {}  # tool calls in progress, by number

    @classmethod
    def http(
        cls,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        auth: "httpx2.Auth | None" = None,
        **options: Any,
    ) -> Self:
        """A remote server, over Streamable HTTP (or a local one at `http://localhost:...`).

        `headers` are sent with every request (`{"Authorization": f"Bearer {token}"}`). `auth` is an
        `httpx2.Auth`, such as the `mcp` package's `OAuthClientProvider` for servers that sign in
        with OAuth. Other options are those of `McpTools(...)`.
        """
        server = url if headers is None and auth is None else _Http(url, headers, auth)
        return cls(server, **options)

    @classmethod
    def stdio(
        cls,
        command: str,
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        **options: Any,
    ) -> Self:
        """A local server: starts `command` with `args` and talks to it over stdin/stdout.

        The process only gets a minimal environment; add what it needs with `env`.
        Other options are those of `McpTools(...)`.
        """
        parameters = mcp.StdioServerParameters(command=command, args=list(args), env=dict(env or {}) or None, cwd=cwd)
        return cls(parameters, **options)

    async def __aenter__(self) -> Self:
        async with AsyncExitStack() as exit_stack:
            server = self._server
            if isinstance(server, _Http):
                # The same timeouts the `mcp` package uses: tool calls may take minutes.
                timeout = httpx2.Timeout(30.0, read=300.0)
                http_client = httpx2.AsyncClient(headers=server.headers, auth=server.auth, timeout=timeout)
                await exit_stack.enter_async_context(http_client)
                server = streamable_http_client(server.url, http_client=http_client)
            options: dict[str, Any] = {"elicitation_callback": self._on_elicitation, **self._client_options}
            self._client = await exit_stack.enter_async_context(mcp.Client(server, **options))
            await self.refresh()
            self._exit_stack = exit_stack.pop_all()  # keep it all open until `__aexit__`
        return self

    async def __aexit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        exit_stack, self._exit_stack, self._client = self._exit_stack, None, None
        if exit_stack is not None:
            await exit_stack.__aexit__(error_type, error, traceback)

    @property
    def client(self) -> mcp.Client:
        """The connected `mcp.Client`, for anything this class does not wrap."""
        if self._client is None:
            raise McpError("Not connected to the MCP server: use it inside `async with McpTools(...)`")
        return self._client

    @property
    def tools(self) -> tuple[McpTool, ...]:
        """The server's tools, listed when connecting (or by the last `refresh()`)."""
        if self._tools is None:
            raise McpError("The tools are listed when connecting: use `async with McpTools(...) as server`")
        return self._tools

    async def refresh(self) -> tuple[McpTool, ...]:
        """Lists the tools again, for servers whose tools change. Agents made before keep the old list."""
        listed: list[mcp_types.Tool] = []
        cursor = None
        while True:
            page = await self.client.list_tools(cursor=cursor)
            listed.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                break
        wanted = [info for info in listed if self._wants(info.name)]
        self._tools = tuple(self.make_tool(info) for info in wanted)
        return self._tools

    def _wants(self, name: str) -> bool:
        return (self._include is None or name in self._include) and name not in self._exclude

    def make_tool(self, info: mcp_types.Tool) -> McpTool:
        """Turns one listed tool into an agent tool. Override to rename tools or change descriptions."""
        name = f"{self._prefix}_{info.name}" if self._prefix else info.name
        parameters = {key: value for key, value in info.input_schema.items() if key != "$schema"}
        parameters.setdefault("properties", {})  # OpenAI wants them even when there are none
        return McpTool(
            # OpenAI accepts letters, digits, `_` and `-`, up to 64 characters; MCP names may have dots.
            name=re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64],
            description=info.description or info.title or "",
            parameters=parameters,
            requires_approval=self._approval_for(info),
            info=info,
            server=self,
        )

    def _approval_for(self, info: mcp_types.Tool) -> RequiresApproval:
        policy = self._requires_approval
        if policy == "unless_read_only":
            return not (info.annotations is not None and info.annotations.read_only_hint)
        if isinstance(policy, bool):
            return policy
        if isinstance(policy, Collection):
            return info.name in policy
        return policy(info)

    async def call_tool(
        self, name: str, arguments: dict[str, Any], ctx: RunContext | None = None
    ) -> ToolContent | ToolReturn:
        """Calls the tool `name` (as the server names it) and converts its result for the model.

        With `ctx`, questions the server asks during the call go to the person (see `elicit`).
        """
        client = self.client
        if ctx is None:
            return self.convert_result(await client.call_tool(name, arguments))
        number = next(_call_numbers)
        call = self._calls[number] = _ToolCall(ctx)
        token = _current_call.set(number)
        try:
            result = await client.call_tool(name, arguments)
        except Exception:
            if call.stop is not None:
                raise call.stop from None  # the server gave up because the question has to wait
            raise
        finally:
            _current_call.reset(token)
            del self._calls[number]
        if call.stop is not None:
            raise call.stop
        return self.convert_result(result)

    def convert_result(self, result: mcp_types.CallToolResult) -> ToolContent | ToolReturn:
        """Turns a tool result into what the model reads (`content`) and what the app gets (`metadata`).

        Text stays text and images become image parts; other content becomes a short text.
        `structuredContent` and the result's `_meta` go to `ToolFinished.metadata`, never to the model.
        An error result raises `McpToolError`, so the model reads it as a tool error.
        """
        parts: list[str | ContentPart] = [_content_part(block) for block in result.content]
        if result.is_error:
            raise McpToolError("\n".join(part for part in parts if isinstance(part, str)) or "The tool failed")
        if not parts and result.structured_content is not None:
            parts = [json.dumps(result.structured_content)]
        texts = [part for part in parts if isinstance(part, str)]
        content: ToolContent = "\n".join(texts) if len(texts) == len(parts) else parts
        metadata = {}
        if result.structured_content is not None:
            metadata["structured_content"] = result.structured_content
        meta = _own_meta(result.meta)
        if meta:
            metadata["meta"] = meta
        return ToolReturn(content, metadata or None)

    async def _on_elicitation(
        self, context: ClientRequestContext, params: mcp_types.ElicitRequestParams
    ) -> mcp_types.ElicitResult:
        """The server asks the user something: find the tool call it belongs to, and ask its run."""
        call = self._calls.get(_current_call.get())
        if call is None and len(self._calls) == 1:
            # Before MCP 2026-07-28, stdio servers ask from another task, so the context does not
            # tell us the call. It is clear when only one call is in progress.
            [call] = self._calls.values()
        if call is None:
            return mcp_types.ElicitResult(action="decline")  # no person to ask (or no way to know whose call)
        # New servers name each question; old ones number them per connection, which changes on resume.
        key = f"mcp:{context.request_id}" if isinstance(context.request_id, str) else None
        try:
            return await self.elicit(call.ctx, params, key=key)
        except StopToolCall as stop:
            call.stop = stop  # `call_tool` raises it once the server has given up
            return mcp_types.ElicitResult(action="cancel")

    async def elicit(
        self, ctx: RunContext, params: mcp_types.ElicitRequestParams, *, key: str | None
    ) -> mcp_types.ElicitResult:
        """Answers a question from the server by asking the person with `ctx.ask` (see the module docs).

        Override it to answer some questions yourself, or to show them differently. `key` matches
        answers to questions when a stopped run resumes.
        """
        data: dict[str, Any] = {
            "type": "mcp_elicitation",
            "server": self._server_name(),
            "mode": params.mode,
            "message": params.message,
        }
        if isinstance(params, mcp_types.ElicitRequestURLParams):
            data["url"] = params.url
        else:
            data["schema"] = params.requested_schema
        answer = await ctx.ask(data, key=key)
        content = None if isinstance(params, mcp_types.ElicitRequestURLParams) else answer.value
        return mcp_types.ElicitResult(action="accept", content=content)

    def _server_name(self) -> str | None:
        info = self._client.server_info if self._client is not None else None
        return info.name if info is not None else self._prefix

    async def prompt(self, name: str, **arguments: str) -> list[Message]:
        """A prompt of the server, as messages for `agent.run(history=...)` or to build on."""
        result = await self.client.get_prompt(name, arguments)
        messages: list[Message] = []
        for prompt_message in result.messages:
            part = _content_part(prompt_message.content)
            if prompt_message.role == "assistant":
                text = part if isinstance(part, str) else json.dumps(part)
                messages.append({"role": "assistant", "content": text})
            else:
                messages.append({"role": "user", "content": part if isinstance(part, str) else [part]})
        return messages

    async def resource(self, uri: str) -> list[str | ContentPart]:
        """The contents of a resource: text, or image and file parts. Put them in a prompt as they are."""
        result = await self.client.read_resource(uri)
        return [_resource_part(contents) for contents in result.contents]


def _content_part(block: Any) -> str | ContentPart:
    """One MCP content block as text or a content part. Anything the model can't take becomes text."""
    if isinstance(block, mcp_types.TextContent):
        return block.text
    if isinstance(block, mcp_types.ImageContent):
        return {"type": "image", "url": f"data:{block.mime_type};base64,{block.data}"}
    if isinstance(block, mcp_types.AudioContent):
        return f"[audio ({block.mime_type}), not shown]"  # tool results cannot carry audio
    if isinstance(block, mcp_types.ResourceLink):
        return f"Resource {block.name}: {block.uri}"
    if isinstance(block, mcp_types.EmbeddedResource):
        return _resource_part(block.resource)
    return json.dumps(block.model_dump(mode="json", by_alias=True, exclude_none=True))


def _resource_part(contents: Any) -> str | ContentPart:
    if isinstance(contents, mcp_types.TextResourceContents):
        return contents.text
    media_type = contents.mime_type or "application/octet-stream"
    url = f"data:{media_type};base64,{contents.blob}"
    if media_type.startswith("image/"):
        return {"type": "image", "url": url}
    filename = str(contents.uri).rstrip("/").rsplit("/", 1)[-1] or "file"
    return {"type": "file", "url": url, "filename": filename}


def _own_meta(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """The server's own `_meta` keys, without the protocol's (`io.modelcontextprotocol/...`)."""
    return {key: value for key, value in (meta or {}).items() if not key.startswith("io.modelcontextprotocol/")}


__all__ = ["ApprovalPolicy", "McpError", "McpTool", "McpToolError", "McpTools"]
