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

Documentation: **[daniprol.github.io/pyox-agents](https://daniprol.github.io/pyox-agents/)**.
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
| `RunFinished`  | `text`, `turns`, `input_tokens`, `output_tokens`, `messages`, `new_messages` |

They work with `match` too: `case TextDelta(text=text): ...`

Models stream text in tiny pieces. To get fewer, bigger `TextDelta` events (less work for Python), choose how
text is grouped with `text_chunks`, on the agent or per run:

| `text_chunks`        | One `TextDelta` per...                                             |
|----------------------|--------------------------------------------------------------------|
| `"delta"` (default)  | piece sent by the API (lowest latency)                             |
| `10` (any number N)  | N pieces                                                           |
| `"paragraph"`        | markdown paragraph; fenced code blocks are never split             |
| `"message"`          | whole assistant message (text before a tool call comes before it)  |

```python
agent = Agent(model="gpt-5-mini", text_chunks="paragraph")
async for event in agent.execute("Explain asyncio", text_chunks=20):  # override for one run
    ...
```

`await agent.run(prompt)` runs to the end and returns the `RunFinished` event.

One `Agent` can serve many runs at the same time: each run has its own history.

## History

Messages are plain dicts (typed as `UserMessage`, `AssistantMessage` and `ToolMessage`), and the same format
goes in and out:

```python
{"role": "user", "content": "Hi"}
{"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "add", "arguments": {"a": 1}}]}
{"role": "tool", "tool_call_id": "call_1", "name": "add", "content": "3", "is_error": False}
```

**Pass it directly.** `history=` takes the messages of a previous run, or JSON text of them (`str` or `bytes`),
which is parsed in Rust without building dicts:

```python
first = await agent.run("I am Dani")
await agent.run("What is my name?", history=first.messages)
await agent.run("What is my name?", history=first.messages_json())  # e.g. read from a database
```

`RunFinished.new_messages` (and `new_messages_json()`) hold only what the run added, to append to your storage.

**Or let the agent load and save it** with a history store. Give it to the agent once, and name the
conversation on each run:

```python
class PostgresHistory:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool  # created once at startup, shared by all runs

    async def load(self, conversation_id: str) -> str:
        # Return message dicts, or JSON text (faster: parsed in Rust).
        return await self.pool.fetchval(
            "SELECT coalesce(jsonb_agg(message ORDER BY position), '[]')::text FROM messages WHERE conversation_id = $1",
            conversation_id,
        )

    async def save(self, conversation_id: str, new_messages_json: str) -> None:  # optional
        ...  # append the run's new messages (JSON text of a list)


agent = Agent(model="gpt-5-mini", history_store=PostgresHistory(pool))
await agent.run("Hi again", conversation_id="user-42")
```

`load` and `save` can be sync or async; a plain `load(conversation_id)` function works too. Failed runs are not
saved. Store errors are raised from the run (`HistoryError` when the store returns something invalid).

**History stores in Rust.** Implement the `HistoryStore` trait from [`pyox-tools`](crates/pyox-tools) on a struct
holding your connection pool (sqlx, diesel, SeaORM...). The agent calls it directly in Rust, so loading never
goes through Python. See [`examples/rust_history_store`](examples/rust_history_store) for a complete SQLite store.

## Images and files

A prompt can be a list of text, images and files (such as PDFs). Paths and bytes are sent inline (base64); a web
address is sent as is.

```python
from pyox_agents import file, image

await agent.run(["What is in this picture?", image("cat.png")])
await agent.run(["Summarize this report", file("report.pdf")])
```

They stay in the history as parts of the user message: `{"type": "image", "url": ...}` and
`{"type": "file", "url": ..., "filename": ...}`.

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

**Run context.** A tool whose first parameter is typed `RunContext` gets the run's context; the model never sees
that parameter. Pass what your tools need (a database pool, the current user...) as `deps`:

```python
from pyox_agents import RunContext


@tool
async def my_orders(ctx: RunContext[Database], status: str = "open") -> list[str]:
    """List the user's orders."""
    return await ctx.deps.orders(ctx.conversation_id, status)


await agent.run("What did I order?", deps=database)
```

`ctx` has `deps`, `conversation_id`, `turn`, `input_tokens` and `output_tokens` (used so far), and, inside a
tool, `tool_call_id` and `tool_name`. Hooks get the same `ctx`.

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

## Hooks

Hooks are functions the agent calls at fixed points of every run, to watch it or change what happens next.
Return `None` to leave things as they are, or a new value to replace them:

| Hook                                   | When                               | Return a new...                         |
|----------------------------------------|------------------------------------|-----------------------------------------|
| `before_run(ctx)`                      | a run starts                       | -                                       |
| `before_model_request(ctx, messages)`  | before each model request          | list of messages, for this request only |
| `after_model_response(ctx, message)`   | after each model response          | assistant message                       |
| `before_tool_call(ctx, call)`          | before each tool call              | call (e.g. other `arguments`)           |
| `after_tool_call(ctx, call, result)`   | after each tool call               | result (`{"content", "is_error"}`)      |
| `after_run(ctx, result)`               | the run finished (`RunFinished`)   | -                                       |

Raising in `before_tool_call` or `after_tool_call` blocks the call: the model gets the error as the tool result.
Raising in any other hook stops the run. Changes made in `before_model_request` don't touch the history, so they
are a safe place to trim long conversations or add reminders.

```python
from pyox_agents import Hooks, RunContext, RunFinished, ToolCall


def no_deletes(ctx: RunContext, call: ToolCall) -> None:
    if call["name"] == "delete_file":
        raise PermissionError("Deleting files is not allowed")


class Timer:  # any object with hook methods works too
    def before_run(self, ctx: RunContext) -> None: ...
    async def after_run(self, ctx: RunContext, result: RunFinished) -> None: ...


agent = Agent(model="gpt-5-mini", tools=[...], hooks=[Hooks(before_tool_call=no_deletes), Timer()])
```

Hooks can be sync or async, and run in the order given: each one sees the changes of the previous ones.

**Hooks in Rust.** Register closures with `Hooks` from [`pyox-tools`](crates/pyox-tools/src/hooks.rs); the agent
calls them directly in Rust, without Python:

```rust
let guard = Hooks::new().before_tool_call(|_context, call| {
    if call.name == "delete_file" {
        return Err("Deleting files is not allowed".into()); // the model sees this
    }
    Ok(())
});
add_hooks(module, "guard", guard)?; // Python: Agent(hooks=[your_package.guard])
```

See `guard` in [`examples/rust_tool_package`](examples/rust_tool_package/src/lib.rs) for a complete example.

## Client

```python
from pyox_agents import Agent, Client

client = Client(
    api_key="sk-...",  # default: $OPENAI_API_KEY
    base_url="https://api.openai.com/v1",  # default: $OPENAI_BASE_URL or OpenAI
    headers={"X-Application": "support"},
    connect_timeout_seconds=10,
    request_timeout_seconds=600,  # covers the whole streamed answer
    max_retries=2,  # resend requests that failed before any response
)
agent = Agent(client=client, model="gpt-5-mini", model_parameters={"temperature": 0.2})
```

Share one `Client` between agents to reuse its connection pool.

A request that fails before any response arrives (for example on a pooled connection the server had already
closed) is sent again, up to `max_retries` times. Once the answer starts streaming it is never retried.

## Errors

All exceptions inherit from `AgentError`:
`ApiError` (with `AuthenticationError` and `RateLimitError`), `RequestTimeoutError`, `NetworkError`,
`InvalidResponseError`, `MaxTurnsExceededError` (the `max_turns` limit, 20 by default), `HistoryError` and
`HookError` (a Rust hook failed, or a hook returned something invalid).

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

The documentation site lives in [`docs/`](docs) and [`zensical.toml`](zensical.toml), built with
[Zensical](https://zensical.org). Preview it with `uv run --only-group docs zensical serve`; every push to `main`
publishes it to GitHub Pages ([`.github/workflows/docs.yml`](.github/workflows/docs.yml)).

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
| Long streamed answer         | 63 ms / 125 ms               | 21 / 7 runs/s           | 61 / 119 MB  |
| One tool call                | 5.1 ms / 21.1 ms             | 354 / 47 runs/s         | 34 / 103 MB  |
| Five tool calls at once      | 6.8 ms / 30.8 ms             | 217 / 30 runs/s         | 36 / 108 MB  |
| Startup (import + agent)     | 83 ms / 1124 ms              |                         | 24 / 76 MB   |

Grouping streamed text makes the long answer much cheaper: 65 ms with `text_chunks="delta"` (default), 9 ms with
`10`, 4.4 ms with `"paragraph"` and 4.9 ms with `"message"`.

Behind a FastAPI streaming endpoint (one uvicorn process, 100 concurrent users, load from [k6](https://k6.io)):

| Scenario             | Requests/s (pyox / pydantic-ai) | First byte p50     | Latency p99       | Server memory |
|----------------------|--------------------------------:|-------------------:|------------------:|--------------:|
| Long streamed answer | 24 / 5                          | 25 ms / 451 ms     | 6.7 s / 20.5 s    | 86 / 120 MB   |
| One tool call        | 339 / 36                        | 102 ms / 1.5 s     | 0.63 s / 3.4 s    | 65 / 122 MB   |

## Next steps

- Let tools return images (e.g. a screenshot tool), as pi does.
- More APIs and providers (Responses, Anthropic, Gemini), each as a new adapter next to `openai_chat.rs`.
- Give Rust tools the run context too (`conversation_id`, `turn`), if a use case needs it.
