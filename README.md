# pyox-agents

Minimal and fast AI agents for Python, with a Rust core. Inspired by [pi](https://github.com/earendil-works/pi).

You define the agent (model, prompt, tools) in Python and iterate over its events.
Rust does the heavy lifting: HTTP, streaming (SSE) decoding, JSON parsing and the agent loop.

```python
import asyncio
from pyox_agents import Agent, TextDelta, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"Sunny in {city}"


agent = Agent(model="gpt-5-mini", prompt="Be concise.", tools=[get_weather])


async def main() -> None:
    async for event in agent.execute("Weather in Vigo?"):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)


asyncio.run(main())
```

Status: early. Only the OpenAI Chat Completions API (and compatible servers) is supported for now.

## Install (development)

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain ([rustup](https://rustup.rs/)).

```bash
uv sync          # builds the Rust extension and installs dev tools
uv run pytest
```

After changing Rust code, `uv sync` (or `uv run ...`) rebuilds it automatically.

## Events

`agent.execute(...)` returns a `Run` you iterate with `async for`. Events are small, read-only objects backed by Rust:

| Event          | Fields                                                              |
|----------------|---------------------------------------------------------------------|
| `TextDelta`    | `text`                                                              |
| `ToolStarted`  | `call_id`, `name`, `arguments` (dict)                               |
| `ToolFinished` | `call_id`, `name`, `result`, `is_error`                             |
| `RunFinished`  | `text`, `turns`, `input_tokens`, `output_tokens`, `messages`        |

They work with `match` too: `case TextDelta(text=text): ...`

`await agent.run(prompt)` runs to the end and returns the `RunFinished` event.
To continue a conversation, pass the previous messages: `agent.run("And tomorrow?", history=previous.messages)`.

One `Agent` can serve many runs at the same time: each run has its own history.

## Tools

**Python tools.** Decorate a function (sync or async) with `@tool`. The JSON schema comes from the type hints,
and the description from the docstring. Plain functions passed in `tools=[...]` are wrapped automatically.

```python
from typing import Annotated, Literal


@tool
async def search(query: Annotated[str, "What to look for"], order: Literal["new", "old"] = "new") -> list[str]:
    """Search the documents."""
```

Results that are not `str` are sent to the model as JSON. If a tool raises, the model gets the error message
(and `ToolFinished.is_error` is `True`); the run continues.

**Built-in Rust tools.** Compiled into the library and run entirely in Rust:

```python
from pyox_agents import BuiltinTool

agent = Agent(model="gpt-5-mini", tools=[BuiltinTool("read_file"), BuiltinTool("write_file")])
BuiltinTool.available()  # ['read_file', 'write_file']
```

Add one in [`src/tools/builtin.rs`](src/tools/builtin.rs): write a function and add it to `BUILTIN_TOOLS`.

**Your own Rust tools, compiled separately.** Write a normal PyO3 package and wrap its functions with `Tool`.
See [`examples/rust_tool_package`](examples/rust_tool_package) and [`examples/rust_tools.py`](examples/rust_tools.py):

```python
import fast_tools
from pyox_agents import Tool

count_words = Tool(
    name="count_words",
    description="Count the words in a text.",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    function=fast_tools.count_words,
    run_in_thread=True,  # keep heavy work off the event loop
)
```

## Client

```python
from pyox_agents import Agent, Client

client = Client(
    api_key="sk-...",  # default: $OPENAI_API_KEY
    base_url="https://api.openai.com/v1",  # default: $OPENAI_BASE_URL or OpenAI
    headers={"X-Application": "support"},
    connect_timeout_seconds=10,
    request_timeout_seconds=600,  # covers the whole streamed answer
)
agent = Agent(client=client, model="gpt-5-mini", model_parameters={"temperature": 0.2})
```

Share one `Client` between agents to reuse its connection pool.

## Errors

All exceptions inherit from `AgentError`:
`ApiError` (with `AuthenticationError` and `RateLimitError`), `RequestTimeoutError`, `NetworkError`,
`InvalidResponseError` and `MaxTurnsExceededError` (the `max_turns` limit, 20 by default).

## How it works

```text
Python                               Rust (src/)
------                               -----------
Agent(...)  ───────────────────────► Engine         config shared by all runs (agent.rs)
agent.execute("...")  ─────────────► Run            one run: history + event queue
async for event in run  ◄─ events ── agent loop     request → stream → tools → repeat
                                       ├── openai_chat.rs   request body + chunk parsing
                                       ├── sse.rs           server-sent events decoding
                                       └── tools/           Python tools are awaited on
                                                            Python's event loop; built-in
                                                            tools run on a Rust thread
```

The Rust code is written to be read by Python developers: small files, descriptive names and comments
explaining the Rust-specific parts. Start with [`src/lib.rs`](src/lib.rs).

## Development

```bash
uv run pytest                 # Python tests (against a local fake OpenAI server)
uv run ruff check . && uv run ruff format .
uv run ty check
cargo test                    # Rust unit tests
cargo clippy --all-targets && cargo fmt
```

## Next steps

- Hooks (before/after tool calls and model requests) that can modify arguments and history.
- More APIs and providers (Responses, Anthropic, Gemini), each as a new adapter next to `openai_chat.rs`.
- Benchmarks against pure-Python agent libraries.
