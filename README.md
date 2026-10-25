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

Examples and the commands to run them are in [`examples/`](examples).

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

Add one in [`src/tools/builtin.rs`](src/tools/builtin.rs).

**Your own Rust tools, compiled separately.** Describe the arguments with a Rust struct and implement the
`RustTool` trait from the [`pyox-tools`](crates/pyox-tools) crate. The JSON schema is generated from the struct
(doc comments become descriptions), so nothing is repeated in Python:

```rust
/// Count the words in a text.
#[derive(Deserialize, JsonSchema)]
struct CountWords {
    /// The text to count words in.
    text: String,
}

impl RustTool for CountWords {
    const NAME: &'static str = "count_words";

    fn run(self) -> Result<String, String> {
        Ok(self.text.split_whitespace().count().to_string())
    }
}
```

```python
import fast_tools
agent = Agent(model="gpt-5-mini", tools=[fast_tools.count_words])
```

The agent calls these Rust functions directly, like the built-in tools: no Python is involved when the model
calls them (see [`crates/pyox-tools/src/native.rs`](crates/pyox-tools/src/native.rs) for how).
See [`examples/rust_tool_package`](examples/rust_tool_package) for a complete package. Built-in tools use the same
trait. Any object with `name`, `description`, `parameters` and an `invoke(arguments)` method (sync or async) works
as a tool too.

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
                                                            Python's event loop; Rust tools
                                                            (built-in or from your own
                                                            package) run on a Rust thread
```

The Rust code is written to be read by Python developers: small files, descriptive names and comments
explaining the Rust-specific parts. Start with [`src/lib.rs`](src/lib.rs).

## Development

```bash
uv sync                       # build the Rust extension and install the dev tools
uv run prek install           # install the git hooks (once)
```

Git hooks ([prek](https://prek.j178.dev), configured in [`prek.toml`](prek.toml)):

| When       | What runs                                                         |
|------------|-------------------------------------------------------------------|
| `git commit` | `ruff check --fix`, `ruff format`, `cargo fmt`, basic file checks |
| `git push`   | `ty check`, `cargo clippy`, `cargo test`, `pytest`              |

Run them by hand with `uv run prek run --all-files` (add `--stage pre-push` for the push hooks).

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs both hook stages on Python 3.11 and 3.14, plus a
coverage job. The coverage report (Python and Rust, including the Rust code exercised by the Python tests) is
shown in the run summary and uploaded as an artifact.

Tests talk to a real local HTTP server that imitates the OpenAI API ([`tests/fake_openai.py`](tests/fake_openai.py)),
so there is no mocking. Build responses with the factories in [`tests/factories.py`](tests/factories.py)
and create agents with the `make_agent` fixture:

```python
async def test_example(fake_openai, make_agent):
    fake_openai.queue(tool_calls_response(tool_call("add", a=1, b=2)), text_response("3"))
    result = await make_agent(tools=[add]).run("1 + 2?")
    assert result.text == "3"
```

## Benchmarks

Compared with [pydantic-ai](https://ai.pydantic.dev) against the same fake OpenAI server, which answers instantly,
so this is the overhead of each library (details and how to run them in [`benchmarks/`](benchmarks)):

| Scenario                     | Latency (pyox / pydantic-ai) | Throughput, 100 at once | Peak memory  |
|------------------------------|-----------------------------:|------------------------:|-------------:|
| Long streamed answer         | 56 ms / 138 ms               | 23 / 7 runs/s           | 56 / 117 MB  |
| One tool call                | 4.3 ms / 20.5 ms             | 434 / 52 runs/s         | 33 / 104 MB  |
| Five tool calls at once      | 5.4 ms / 31.3 ms             | 272 / 31 runs/s         | 35 / 107 MB  |
| Startup (import + agent)     | 73 ms / 1079 ms              |                         | 23 / 76 MB   |

## Next steps

- Hooks (before/after tool calls and model requests) that can modify arguments and history.
- More APIs and providers (Responses, Anthropic, Gemini), each as a new adapter next to `openai_chat.rs`.
