# Changelog

All notable changes to pyox-agents (and the `pyox-tools` crate). The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Before 1.0, a minor version may break things; each
break comes with a migration note.

## [Unreleased]

### Added

- MCP servers: `pyox_agents.mcp.McpTools` gives an agent the tools of any MCP server, over Streamable HTTP
  (`McpTools.http(url, headers=...)`) or stdio (`McpTools.stdio(command, args)`). Tools keep the server's names,
  descriptions and schemas (`prefix=`, `include=`, `exclude=` to choose), can need approval
  (`requires_approval="unless_read_only"`, ...), and send images and structured results. When the server asks
  the user something, the question becomes an `InputRequested`, answered live or after the run stops. Works with
  servers on MCP 2025-11-25 and 2026-07-28. Install with `pip install "pyox-agents[mcp]"`.
- `run(tools=...)` and `execute(tools=...)` add tools for one run, on top of the agent's: for tools that depend on
  the user, such as an MCP server opened with their token.

### Fixed

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
