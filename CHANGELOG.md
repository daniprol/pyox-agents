# Changelog

All notable changes to pyox-agents (and the `pyox-tools` crate) are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org)
(before 1.0, a minor version may break things: each break comes with a migration note).

## [Unreleased]

## [0.2.0] - 2026-11-01

### Breaking changes

- **`Agent(prompt=...)` is now `Agent(instructions=...)`**, the name both OpenAI APIs use.
  Migrate: rename the keyword argument.
- **`before_model_request(ctx, messages)` is now `before_model_request(ctx, request)`**. `request` is a
  `ModelRequest` dict with `instructions`, `messages`, `tools` (names) and `model_parameters`, so one hook can
  change all of them. Migrate: read `request["messages"]`, and return `{**request, "messages": new_messages}`
  instead of the list (or `None` for no change).
- **`SubagentFinished.waiting` is gone**: use `SubagentFinished.stop_reason == "waiting"` (`None` when the
  subagent failed).
- **`Agent` settings are read-only** (`model`, `instructions`, `api`, `tools`, `output_type`). Setting them
  silently did nothing before. Migrate: create a new `Agent`.
- **The built-in `read_file` and `write_file` tools (`BuiltinTool`) are removed.** They were demos with no
  sandboxing. Migrate: copy them from [`examples/rust_tool_package`](examples/rust_tool_package), where they now
  live as an example, or write your own tool.
- **Tool schemas are more precise**: `X | None` parameters allow `null` (`anyOf` with `{"type": "null"}`), and
  `Literal` values get their `type`. Migrate: only tests that compare whole schemas change.
- **`Client(max_retries=...)` also retries 429, 408, 409 and 5xx answers** (it only retried network failures
  before), waiting as the server asks. Migrate: pass `max_retries=0` to handle them yourself.
- **`pyox-tools` 0.2.0 (Rust)**: Rust hooks get `before_model_request(|context, request| ...)` with a
  `&mut ModelRequest` instead of the messages; `UserContent` is now `Content` (also the type of tool message and
  `ToolResult` content, so results can hold images); `success_result` takes anything `Into<Content>`. The hook
  interface version is now 3. Migrate: update the crate and rebuild your Rust packages; an old build is
  rejected with a clear error.

### Added

- Tracing with OpenTelemetry: `pyox_agents.instrument()` or `Agent(instrument=...)`, with the optional extra
  `pyox-agents[otel]` (the OpenTelemetry API only; your app sets up the SDK and exporters). Runs, model requests
  and tool calls become `invoke_agent`, `chat` and `execute_tool` spans (GenAI semantic conventions, experimental
  names), subagents nest under their tool call, and your own spans in tools and hooks nest where they ran. Also
  the `gen_ai.client.*` token, duration and time-to-first-chunk metrics. Off by default; prompts and results only
  with `capture_content=True` (or `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`). See docs/telemetry.md.
- `Agent(name=...)`: names the agent in traces.
- `RunFinished.stop_reason`: why the run ended (`"end"`, `"waiting"`, `"length"`, `"refusal"` or `"stopped"`).
- `run.stop()`: ends a run at the next safe point, keeping the text streamed so far and a valid history.
- `error.messages` on errors raised by a run: the history up to the failure, to continue from.
  `MaxTurnsExceededError` has it too.
- Structured output: `Agent(output_type=...)` with a dataclass, `TypedDict`, pydantic model (used by duck
  typing, pydantic is not a dependency) or JSON schema, sent in OpenAI's strict mode with both APIs.
  `RunFinished.output` has the answer as that type. The schema generator is `pyox_agents.schema`.
- Tool parameters can be dataclasses, `TypedDict`s and pydantic models; dataclass and pydantic arguments arrive
  built.
- Tool results can hold images and files (`ToolReturn(["The page:", image(...)])`, or `image(...)` alone).
  The Responses API gets them in the tool output; Chat Completions gets them in a user message after the tool
  messages, since its tool messages only take text. New `ToolFinished.content` and `ToolContent`.
- Retries honor `Retry-After` and `retry-after-ms`, with a growing delay (and jitter) otherwise, never after the
  answer started streaming. Also on the Responses WebSocket transport.
- `ApiError` has `status_code`, `body`, `request_id` and `retry_after_seconds`.
- Responses API built-in tools (web search, file search...) work through `model_parameters["tools"]`.

### Fixed

- Tools given in `model_parameters["tools"]` were replaced by the agent's function tools; they are now kept
  first.
- Sync hooks, tools, history stores and `on_request` handlers ran without the caller's context variables (a
  request id, the current tracing span...). They now run in a copy of the run's context, like async ones.
- Dataclass and `TypedDict` tool parameters got an empty schema (any value).
- A Chat Completions refusal or content filter stop was read as a normal answer.

## [0.1.1] - 2026-10-31

### Added

- Subagents: `agent.as_tool(...)`, with `SubagentStarted` and `SubagentFinished` events, and subagents that ask
  the person or their calling agent.
- Shared state: `execute(state=...)`, `ctx.state` in tools, `StateChanged` events and `RunFinished.state`, also
  as AG-UI and AI SDK events.

## [0.1.0] - 2026-10-30

First release: agents with a Rust core for the OpenAI Chat Completions, Responses and Decisions APIs (and
Azure OpenAI), streaming events, Python and Rust tools, history stores, hooks, human in the loop, images and
files in prompts, and AI SDK and AG-UI web streams.

[Unreleased]: https://github.com/daniprol/pyox-agents/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/daniprol/pyox-agents/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/daniprol/pyox-agents/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/daniprol/pyox-agents/releases/tag/v0.1.0
