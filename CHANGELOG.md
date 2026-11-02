# Changelog

All notable changes to pyox-agents (and the `pyox-tools` crate). The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Before 1.0, a minor version may break things; each
break comes with a migration note.

## [Unreleased]

### Breaking changes

- `agent.execute(...)` is now `agent.stream(...)`, and `agent.execute_sync(...)` is now `agent.stream_sync(...)`.
  They take the same arguments. `run(...)` and `run_sync(...)` still return the finished `RunFinished` without
  streaming. Migrate: rename the calls. `ai_sdk.stream(...)` and `ag_ui.stream(...)` keep their names.
- Errors from your hooks, `on_request` and history stores now raise as `HookError` or `HistoryError`, with your
  exception as `__cause__`, so `except AgentError` catches every failure of a run. Migrate:
  `except PermissionError` around a run whose `before_run` raises it becomes `except HookError as error` and
  `error.__cause__`. Or raise `StopRun("...")` in the hook, which the run raises as it is.
- Tool arguments are checked against the tool's type hints before the call. Before, a tool typed `days: int`
  could get `"3"`; now the model reads `days: expected an integer, got "3"` and the tool is not called.
- `ApiError` messages are short (`HTTP 400: Too long`). The whole body is still in `error.body`.

### Added

- Error handling for tools (see [Errors](https://daniprol.github.io/pyox-agents/errors/)):
  - `raise ToolFailed("...")` in a tool: the model reads exactly that message. Other exceptions are shown as
    `ValueError: ...` and logged as warnings on the `pyox_agents` logger.
  - `Hooks(on_tool_error=...)` and `@tool(on_error=...)` choose what the model reads when a tool fails, with the
    exception itself.
  - `raise StopRun("...")` in a tool or hook ends the run; the run raises it, with a history you can continue.
  - `@tool(timeout_seconds=...)`: the model reads that the call timed out, and an async tool is cancelled.
  - Bad arguments raise `InvalidArguments`: the model reads one short line per problem and decides whether to try
    again. pyox never retries for it.
- `ApiError.code`, OpenAI's error code (like `"context_length_exceeded"`).
- Types: `stream()` returns a `Run[Output]` (and `stream_sync()` a `SyncRun[Output]`), so `RunFinished.output`
  has the agent's output type in every event. `ctx.ask(...)` returns an `Answer` and `ctx.approval` is an
  `Approve` or `None`. A bare `RunContext` means `RunContext[Any]`.
- MCP servers: `pyox_agents.mcp.McpTools` gives an agent the tools of any MCP server, over Streamable HTTP
  (`McpTools.http(url, headers=...)`) or stdio (`McpTools.stdio(command, args)`). Tools keep the server's names,
  descriptions and schemas (`prefix=`, `include=`, `exclude=` to choose), can need approval
  (`requires_approval="unless_read_only"`, ...), and send images and structured results. When the server asks
  the user something, the question becomes an `InputRequested`, answered live or after the run stops. Works with
  servers on MCP 2025-11-25 and 2026-07-28. Install with `pip install "pyox-agents[mcp]"`.
- A sync API, for code without an event loop (scripts, Django and Flask views, Celery tasks, notebooks):
  `agent.run_sync(...)`, and `agent.stream_sync(...)` read with `for` (also `run.batches()`, `run.ai_sdk()` and
  `run.ag_ui(...)`, for Django's `StreamingHttpResponse`). Rust runs the agent with no Python event loop: sync
  tools, hooks and handlers run on your thread, async ones on pyox's own event loop thread. Ctrl+C works. Sync tools
  ask and save steps with `ctx.ask_sync(...)` and `ctx.step_sync(...)`. MCP servers open with `with` for sync runs.
- `run(tools=...)` and `stream(tools=...)` add tools for one run, on top of the agent's: for tools that depend on
  the user, such as an MCP server opened with their token.

### Changed

- Sync tools no longer block the event loop: async runs call them on worker threads, so several run in parallel.
  `Tool.invoke` is now a plain `def` that returns a coroutine for async functions. Migrate: tests that
  `await tool.invoke(...)` on a sync tool call it without `await`.

### Fixed

- A script that ended right after its last run could crash at exit (`Segmentation fault`) on Python 3.11 to
  3.13, about once in 20 exits. Python now waits for pyox to finish handing it results before it shuts down.
- `ToolFinished` comes as soon as its call ends. Before, the results of parallel tool calls all came when the
  slowest call ended.
- `StopRun` in a call resumed with `answers=` keeps the results of the other answered calls in `error.messages`,
  like it does in a new turn.
- Cancelling the task that reads a run (or a timeout) cancels the async tools still running, instead of leaving
  them running.
- A run that fails in `after_run` or while saving its history keeps its messages in `error.messages`.
- `StopRun` in one of several parallel tool calls leaves every call with a result, so the history stays valid.
- A rate limit inside a stream keeps its `request_id`, `NetworkError` says why the connection failed, odd
  `Retry-After` values are ignored, and invalid `Client` timeouts raise a clear `ValueError`.
- A failed subagent becomes its tool's error result, instead of ending the parent run.
- `Usage` and run errors (with their `usage` and `messages`) can be pickled, for Celery and multiprocessing.
- A process forked after pyox-agents ran in its parent (Gunicorn `--preload`, Celery prefork) raises a clear
  `RuntimeError` instead of hanging.
- A run's stream (`async for event in run`, `run.batches()`) ends with its `RunFinished` or error. Before, it hung
  while anything still held a tool's `ctx`, such as a cache or a library's background task.

## [0.2.0] - 2026-11-01

### Breaking changes

- `Agent(prompt=...)` is now `Agent(instructions=...)`. Migrate: rename the argument.
- `before_model_request(ctx, messages)` is now `before_model_request(ctx, request)`, where `request` holds
  `instructions`, `messages`, `tools` and `model_parameters`. Migrate: read `request["messages"]` and return
  `{**request, "messages": new_messages}` (or `None`).
- `Agent` settings (`model`, `instructions`, `api`, `tools`, `output_type`) are read-only; setting them did
  nothing before. Migrate: create a new `Agent`.
- `BuiltinTool` (`read_file`, `write_file`) is removed. Migrate: copy them from
  [`examples/rust_tool_package`](examples/rust_tool_package) or write your own.
- `Client(max_retries=...)` also retries 408, 409, 429 and 5xx answers. Migrate: `max_retries=0` to opt out.
- Tool schemas are more precise: `X | None` allows `null`, and `Literal` values get their `type`.
- `pyox-tools` 0.2.0 (Rust): `before_model_request` gets a `&mut ModelRequest`, `UserContent` is now `Content`,
  and the hook interface is version 3. Migrate: update the crate and rebuild.

### Added

- Subagents: `agent.as_tool(...)`, with `SubagentStarted` and `SubagentFinished` events. Subagents can ask the
  person or their calling agent.
- Shared state: `execute(state=...)`, `ctx.state` in tools, `StateChanged` events and `RunFinished.state`
  (also in AG-UI and AI SDK streams).
- OpenTelemetry tracing: `pyox_agents.instrument()` or `Agent(instrument=...)`, with the `pyox-agents[otel]`
  extra. GenAI spans and metrics, off by default, no prompts recorded unless `capture_content=True`. See
  [Telemetry](https://daniprol.github.io/pyox-agents/telemetry/).
- Structured output: `Agent(output_type=...)` with a dataclass, `TypedDict`, pydantic model or JSON schema;
  the answer is in `RunFinished.output`.
- Dataclass, `TypedDict` and pydantic tool parameters, passed to the tool already built.
- Images and files in tool results (`ToolReturn([...])`, `ToolFinished.content`).
- `run.stop()`, `RunFinished.stop_reason`, and `error.messages` (the history up to a failure).
- `Agent(name=...)`, used in traces.
- Retries honor `Retry-After`, with backoff and jitter otherwise.
- `ApiError.status_code`, `body`, `request_id` and `retry_after_seconds`.
- Responses API built-in tools (web search, file search...) through `model_parameters["tools"]`.

### Fixed

- Tools in `model_parameters["tools"]` were replaced by the agent's tools.
- Sync tools, hooks and history stores ran without the caller's context variables.
- Dataclass and `TypedDict` tool parameters got an empty schema.
- A Chat Completions refusal or content filter stop was read as a normal answer.

## [0.1.0] - 2026-10-30

First release: agents with a Rust core for the OpenAI Chat Completions, Responses and Decisions APIs (and
Azure OpenAI), streaming events, Python and Rust tools, history stores, hooks, human in the loop, images and
files in prompts, and AI SDK and AG-UI web streams.

[Unreleased]: https://github.com/daniprol/pyox-agents/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/daniprol/pyox-agents/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/daniprol/pyox-agents/releases/tag/v0.1.0
