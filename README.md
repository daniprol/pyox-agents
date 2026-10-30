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

Status: early. It supports the OpenAI Chat Completions API (and compatible servers), the Responses API, the Decisions API, and Azure OpenAI.

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
| `RunFinished`  | `text`, `turns`, `usage`, `response_id`, `messages`, `new_messages` |

They work with `match` too: `case TextDelta(text=text): ...`

`usage` is a `Usage` with the tokens of the whole run: `input_tokens`, `cached_input_tokens`, `output_tokens`,
`reasoning_tokens` and `requests`. Counts the API did not report are 0. Add them up with `+`, and read
`error.usage` on an `AgentError` to see what a failed run used.

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

## Responses API

`api="responses"` uses OpenAI's Responses API. The model's encrypted reasoning is kept in the history and sent back
unchanged, so the model builds on it and the prompt cache keeps working. With `store=True`, OpenAI keeps each
response and requests send only the new messages (`previous_response_id`), inside a run and across runs:

```python
agent = Agent(model="gpt-5-mini", api="responses", store=True)
first = await agent.run("I am Dani")
await agent.run("What is my name?", previous_response_id=first.response_id)
```

**WebSocket mode.** With `Client(transport="websocket")`, Responses API requests travel over WebSocket
connections that Rust pools per `Client`. A run keeps one connection for all its turns, so after the first turn
each request sends only the new tool results, even without `store` (the server caches the latest response of each
connection). Chat Completions requests stay on HTTP, and a server without WebSocket support gets HTTP requests.

Details in the [Responses API docs](https://daniprol.github.io/pyox-agents/responses-api/).

## Decisions API

OpenAI's Decisions API answers questions about an input with probabilities, fast and cheap: handy for triage,
routing and guard rails around an agent. The input is text, a prompt with images, or a whole history:

```python
from pyox_agents import Choice, Predicate, Score, decide

decision = await decide(
    "The deploy failed twice and customers are seeing 500s.",
    {
        "urgent": Predicate("Does this need attention right now?"),
        "team": Choice("Which team should handle it?", {"infra": "Deploys and outages.", "billing": "Payments."}),
        "severity": Score("How severe is the impact?", ["Cosmetic", "Degraded", "Outage"]),
    },
    client=client,
)
decision["urgent"].probability  # 0.97
decision["team"].choice         # "infra"
```

Any answer can be a `Refusal` instead. Details and recipes (approval for risky tool calls, picking an agent) in the
[Decisions API docs](https://daniprol.github.io/pyox-agents/decisions/).

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

`ctx` has `deps`, `conversation_id`, `turn`, `usage` (tokens used so far), and, inside a
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

## Human in the loop

A tool call can wait for a person: an approval (`@tool(requires_approval=True)`), a question from anywhere in
the tool body (`await ctx.ask(data, metadata=...)`), or a frontend tool your app runs (`@tool(frontend=True)`).
Reply live with `on_request=`, or let the run stop and send the answers in a later run. No database needed:
everything is in the history.

```python
from pyox_agents import Answer, Approve, RunContext, tool


@tool
async def book_flight(ctx: RunContext, city: str) -> str:
    """Book a flight."""
    answer = await ctx.ask({"options": [f"{city} 9:00", f"{city} 18:00"]}, metadata={"widget": "picker"})
    return f"Booked {answer.value}"


result = await agent.run("Fly me to Paris")  # stops: result.pending lists the InputRequested
result = await agent.run(history=result.messages, answers={result.pending[0].call_id: Answer("Paris 18:00")})
```

`metadata` (on requests, replies and `ToolReturn` results) is never sent to the model. A resumed tool runs again
from its start, with earlier answers returned at once; `ctx.step(name, function)` runs a side effect only once.
See [Human in the loop](docs/human-in-the-loop.md) for approvals, `Deny`, frontend tools and unanswered calls.

## Web frontends (AI SDK and AG-UI)

Serve an agent to a chat UI built with the [Vercel AI SDK](https://ai-sdk.dev) (`useChat`, v6+) or
[AG-UI](https://docs.ag-ui.com) (CopilotKit). Rust writes the stream; approvals, questions and frontend tools use
each toolkit's own way of waiting for the user.

```python
from pyox_agents import ag_ui, ai_sdk


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    return StreamingResponse(ai_sdk.stream(agent, await request.body()), headers=ai_sdk.HEADERS)


@app.post("/api/agent")
async def run_agent(request: Request) -> StreamingResponse:
    return StreamingResponse(ag_ui.stream(agent, await request.body()), headers=ag_ui.HEADERS)
```

See [Web frontends](https://daniprol.github.io/pyox-agents/web-frontends/) and
[`examples/web_server.py`](examples/web_server.py).

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
    transport="http",  # or "websocket" for Responses API requests
)
agent = Agent(client=client, model="gpt-5-mini", model_parameters={"temperature": 0.2})
```

Share one `Client` between agents to reuse its connection pool.

A request that fails before any response arrives (for example on a pooled connection the server had already
closed) is sent again, up to `max_retries` times. Once the answer starts streaming it is never retried.

**Azure OpenAI** (Microsoft Foundry, v1 API): `Client.azure("https://my-resource.openai.azure.com", api_key=...)`
(defaults: `$AZURE_OPENAI_ENDPOINT`, `$AZURE_OPENAI_API_KEY`). No `api-version` is needed, and the agent's
`model` is your deployment name.

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
                                       ├── model_api.rs     picks the API adapter
                                       ├── openai_chat.rs   Chat Completions: request body + chunk parsing
                                       ├── openai_responses.rs  Responses: request body + event parsing
                                       ├── sse.rs           server-sent events decoding
                                       ├── ws.rs, ws_pool.rs  Responses over WebSocket: one connection, the pool
                                       └── tools/           Python tools are awaited on
                                                            Python's event loop; Rust tools
                                                            (built-in or from your own
                                                            package) run on a Rust thread
run.ai_sdk(), run.ag_ui(...)  ◄─ text ── run.rs   events in batches → ai_sdk.rs, ag_ui.rs
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

Tests use a local fake OpenAI server. A few live tests in [`tests/test_live.py`](tests/test_live.py) call a
real model on Azure OpenAI instead; they are skipped by default. Run them with
`uv run pytest -m live` after setting `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and optionally
`LIVE_MODEL` (the deployment, `gpt-6-luna` by default). CI runs them on pushes to `main` and `dev`, from
the `live` GitHub environment, with `LIVE_TEST_WORKERS` (in `ci.yml`) tests at a time.

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

Compared with seven other agent frameworks (pydantic-ai, OpenAI Agents SDK, Microsoft Agent Framework, LangGraph,
LangChain, LlamaIndex and Agno) against the same fake OpenAI server, which answers instantly, so this is the
overhead of each library. "Others" is the range from the best to the worst of the seven (details, every number
and how to run them in [`benchmarks/`](benchmarks)):

| Scenario                 | Latency p50 (pyox / others) | Throughput, 100 at once   | Peak memory        |
|--------------------------|----------------------------:|--------------------------:|-------------------:|
| Long streamed answer     | 105 ms / 112-213 ms         | 14 / 4-9 runs/s           | 50 / 100-381 MB    |
| One tool call            | 8.3 ms / 17.6-41.0 ms       | 230 / 23-57 runs/s        | 35 / 94-196 MB     |
| Five tool calls at once  | 10.9 ms / 28.1-65.8 ms      | 147 / 14-37 runs/s        | 36 / 99-204 MB     |
| Startup (import + agent) | 0.13 s / 1.5-3.2 s          |                           | 25 / 67-132 MB     |

Grouping streamed text makes the long answer much cheaper: 111 ms with `text_chunks="delta"` (default), 15 ms with
`10`, 6.7 ms with `"paragraph"` and 7.6 ms with `"message"`.

Behind a FastAPI streaming endpoint (one uvicorn process, 100 concurrent users, load from [k6](https://k6.io)):

| Scenario             | Requests/s (pyox / others) | First byte p50       | Latency avg          | Server memory    |
|----------------------|---------------------------:|---------------------:|---------------------:|-----------------:|
| Long streamed answer | 17 / 3-8                   | 49 ms / 0.9-10.4 s   | 5.6 s / 11.1-28.7 s  | 83 / 105-216 MB  |
| One tool call        | 247 / 24-56                | 118 ms / 1.6-3.3 s   | 0.4 s / 1.7-4.1 s    | 65 / 107-200 MB  |

## Next steps

- Let tools return images (e.g. a screenshot tool), as pi does.
- More APIs and providers (Responses, Anthropic, Gemini), each as a new adapter next to `openai_chat.rs`.
- Give Rust tools the run context too (`conversation_id`, `turn`), if a use case needs it.
