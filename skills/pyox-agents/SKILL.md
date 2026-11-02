---
name: pyox-agents
description: Build AI agents in Python with pyox-agents (Rust core, OpenAI Chat Completions and Responses APIs). Use when writing or reviewing code that imports pyox_agents, when adding tools, hooks, history stores, subagents, human-in-the-loop or web streaming to such an agent, or when writing pyox tools, hooks or history stores in Rust with the pyox-tools crate.
license: MIT
---

# pyox-agents

pyox-agents is a Python library for AI agents whose agent loop (HTTP, streaming, JSON, tool dispatch) runs in
Rust. Python code defines agents and tools; the API is `async`, with sync twins for code without an event loop. Full docs: https://daniprol.github.io/pyox-agents/

## Rules that prevent most mistakes

- `pip install pyox-agents` (or `uv add pyox-agents`); Python 3.11+. No Python dependencies.
- Read `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the environment, or pass a `Client`. One `Client` per process,
  shared by agents (it holds the connection pool). One `Agent` serves many concurrent runs.
- `agent.stream(...)` returns a `Run` to iterate with `async for`; `await agent.run(...)` returns the final
  `RunFinished`. Both take `history=`, `conversation_id=`, `deps=`, `answers=`, `on_request=`, `state=`, and
  `tools=` (extra tools for that run only, e.g. a user's own MCP server).
- Web streams (`run.ai_sdk()`, `run.ag_ui(...)`, `run.batches()`) exist only on the `Run` from `stream()`, never on
  the `RunFinished` from `run()`.
- In sync code (scripts, Django/Flask views, Celery), use `agent.run_sync(...)` and `for event in
  agent.stream_sync(...)` instead; sync tools then use `ctx.ask_sync` / `ctx.step_sync`. Never wrap the async
  API in `asyncio.run` for this.
- Tool errors do not raise: the model reads them and decides what to do. In tools, raise `ToolFailed("message for
  the model")` for expected failures and `StopRun("why")` to end the run. Failed runs raise subclasses of
  `AgentError`, with `error.messages` (history so far) and `error.usage`. Never add retry loops for bad tool
  arguments: the model reads what is wrong and `max_turns` limits it.
- History is plain dicts (`{"role": "user" | "assistant" | "tool", ...}`), the same format in and out.
  Never build OpenAI-format messages by hand.
- Do not invent parameters. If something is not on this page, check the docs page linked for it.

## A complete agent

```python
import asyncio
from pyox_agents import Agent, RunFinished, TextDelta, ToolStarted, tool


@tool
async def get_weather(city: str) -> str:
    """Get the current weather in a city."""  # the docstring is the tool description
    return f"Sunny in {city}"


agent = Agent(model="gpt-5-mini", instructions="Be concise.", tools=[get_weather])


async def main() -> None:
    async for event in agent.stream("Weather in Vigo?"):
        match event:
            case TextDelta(text=text):
                print(text, end="", flush=True)
            case ToolStarted(name=name, arguments=arguments):
                print(f"\n[{name}({arguments})]")
            case RunFinished(stop_reason=reason, usage=usage):
                print(f"\n{reason}, {usage.input_tokens} in / {usage.output_tokens} out")


asyncio.run(main())
```

`Agent(...)` options: `model`, `name`, `instructions`, `tools`, `client`, `model_parameters` (e.g.
`{"temperature": 0.2}`), `max_turns` (20), `text_chunks` (`"delta"`, an int N, `"paragraph"`, `"message"`),
`history_store`, `hooks`, `api` (`"chat"` or `"responses"`), `store`, `output_type`, `instrument`.

`RunFinished` has `text`, `stop_reason` (`"end"`, `"waiting"`, `"length"`, `"refusal"`, `"stopped"`), `output`,
`turns`, `usage`, `response_id`, `messages`, `new_messages`, `pending`.

## Tools

- `@tool` on a sync or async function. Schema from type hints: `str`, `int`, `float`, `bool`, lists, dicts,
  `Literal`, `X | None`, dataclasses, `TypedDict`, pydantic models. `Annotated[str, "description"]` describes a
  parameter. Dataclass and pydantic arguments arrive built. Non-`str` results are sent as JSON.
- First parameter typed `RunContext[Deps]` gets the run context (hidden from the model): `ctx.deps` (whatever
  was passed as `deps=`), `ctx.conversation_id`, `ctx.turn`, `ctx.usage`, `ctx.state`, `ctx.ask(...)`,
  `ctx.step(name, function)`.
- `@tool(requires_approval=True)` (or a function `(ctx, arguments) -> bool`) waits for a person.
  `@tool(frontend=True)` is run by your UI.
- Docs: https://daniprol.github.io/pyox-agents/tools/

## Common tasks

| Task | How | Docs |
|---|---|---|
| Continue a conversation | `agent.run(prompt, history=previous.messages)` or JSON text from `previous.messages_json()` | [history](https://daniprol.github.io/pyox-agents/history/) |
| Persist conversations | `Agent(history_store=obj)` where `obj.load(conversation_id)` returns messages or JSON text, optional `obj.save(conversation_id, new_messages_json)`; then `run(..., conversation_id="...")` | [history](https://daniprol.github.io/pyox-agents/history/#history-stores) |
| Typed answer | `Agent(output_type=MyDataclass)`; `result.output` is an instance (strict mode: optional fields must allow `None`) | [structured output](https://daniprol.github.io/pyox-agents/structured-output/) |
| Approvals and questions | run stops with `stop_reason == "waiting"` and `result.pending`; resume with `agent.run(history=result.messages, answers={call_id: Approve() / Deny(...) / Answer(value)})`, or answer live with `on_request=` | [human in the loop](https://daniprol.github.io/pyox-agents/human-in-the-loop/) |
| Subagents | `writer = Agent(..., tools=[researcher.as_tool("research", "Research a topic", state="read")])` | [subagents](https://daniprol.github.io/pyox-agents/subagents/) |
| Shared state | `stream(state={...})`; tools change `ctx.state`; `StateChanged` events | [state](https://daniprol.github.io/pyox-agents/state/) |
| Handle tool failures | `Hooks(on_tool_error=fn)` or `@tool(on_error=fn)`: `fn(ctx, call, error)` returns the text the model reads, or `None` for the default; `@tool(timeout_seconds=30)` | [errors](https://daniprol.github.io/pyox-agents/errors/) |
| Watch or change a run | `hooks=[Hooks(before_tool_call=fn)]` or any object with hook methods; raise in a tool hook to block the call | [hooks](https://daniprol.github.io/pyox-agents/hooks/) |
| Serve a chat UI | `StreamingResponse(ai_sdk.stream(agent, await request.body()), headers=ai_sdk.HEADERS)`; same with `ag_ui` | [web frontends](https://daniprol.github.io/pyox-agents/web-frontends/) |
| Images and PDFs | `agent.run(["Describe it", image("cat.png")])`, `file("report.pdf")` | [images and files](https://daniprol.github.io/pyox-agents/images-and-files/) |
| Tools of an MCP server | `async with McpTools.http(url) as server: Agent(..., tools=server.tools)` from `pyox_agents.mcp` (`pip install "pyox-agents[mcp]"`); also `McpTools.stdio(command, args)` | [MCP servers](https://daniprol.github.io/pyox-agents/mcp/) |
| Tracing | `pyox_agents.instrument()` after configuring OpenTelemetry (`pip install "pyox-agents[otel]"`) | [tracing](https://daniprol.github.io/pyox-agents/telemetry/) |
| Other servers, Azure, retries | `Client(base_url=..., api_key=..., max_retries=2)`, `Client.azure(endpoint)` | [client](https://daniprol.github.io/pyox-agents/client/) |
| Responses API | `Agent(api="responses")`, optionally `store=True` and `Client(transport="websocket")` | [Responses API](https://daniprol.github.io/pyox-agents/responses-api/) |
| Classify or route cheaply | `await decide(text, {"urgent": Predicate("...")}, client=client)` | [Decisions API](https://daniprol.github.io/pyox-agents/decisions/) |

Hook names (Python and Rust): `before_run`, `before_model_request`, `after_model_response`, `before_tool_call`,
`after_tool_call`, `after_run`. Return `None` to keep things as they are, or a new value to replace them.

## Writing tools, hooks or history stores in Rust

When a tool is CPU-heavy or called very often, write it in Rust in a separate PyO3 package that depends on the
`pyox-tools` crate. The agent calls it without Python. Read [references/rust.md](references/rust.md) before
writing any Rust for pyox-agents.

## Testing agents

Point a `Client(base_url=...)` at a local fake OpenAI-compatible server instead of mocking pyox internals; the
repository's `tests/fake_openai.py` and `tests/factories.py` show one.
