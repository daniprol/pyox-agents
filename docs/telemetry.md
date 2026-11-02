# Tracing (OpenTelemetry)

pyox-agents can trace every run with [OpenTelemetry](https://opentelemetry.io). A run becomes one
`invoke_agent` span, with a `chat` span for each model request (tokens, timing) and an `execute_tool` span for
each tool call. A subagent's run goes under the tool call that runs it, and your own spans inside tools and hooks
nest where they ran:

```text
POST /chat                                  your web framework's span: the run's parent
└─ invoke_agent writer
   ├─ chat gpt-5                            tokens, time to first chunk, finish reason
   ├─ execute_tool research                 a subagent
   │  └─ invoke_agent research
   │     ├─ chat gpt-5-mini
   │     ├─ execute_tool web_search         two calls that ran at the same time
   │     │  └─ GET api.search.example       your httpx span, inside the tool
   │     ├─ execute_tool web_search
   │     └─ chat gpt-5-mini
   └─ chat gpt-5                            the final answer
```

pyox-agents only *emits* spans and metrics, through the OpenTelemetry API, to the providers your app set up.
It never configures exporters, so Logfire, Langfuse, Phoenix, Jaeger, Datadog or any OTLP backend all work the
way they already do in your app, with your sampler and processors.

## Turn it on

```bash
pip install "pyox-agents[otel]"   # adds opentelemetry-api, nothing else
```

```python
import pyox_agents

pyox_agents.instrument()  # trace the runs of every agent
```

Tracing is off until you ask: installing OpenTelemetry alone changes nothing, and runs that are not traced cost
nothing (Rust never calls into Python for them).

Per agent, `Agent(instrument=...)` overrides `instrument()`: `False` never traces, `True` traces with the global
providers, and an `Instrumentation` picks its own:

```python
from pyox_agents import Agent, Instrumentation

agent = Agent(
    model="gpt-5",
    name="triage",  # names the run's span: `invoke_agent triage`
    instrument=Instrumentation(tracer_provider=tenant_provider, meter_provider=tenant_meters),
)
```

`name` is optional: without it the span is just `invoke_agent`, and a subagent is named after its tool.

## Send it somewhere

Set up OpenTelemetry the way your backend says, then call `pyox_agents.instrument()`.

=== "Logfire"

    ```python
    import logfire

    logfire.configure()  # sets the global tracer and meter providers
    pyox_agents.instrument()
    ```

=== "OTLP (Jaeger, Tempo, Honeycomb...)"

    ```python
    # pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "support-bot"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # reads OTEL_EXPORTER_OTLP_*
    trace.set_tracer_provider(provider)
    pyox_agents.instrument()
    ```

    To look at traces on your machine: `docker run --rm -p 4318:4318 -p 16686:16686 jaegertracing/jaeger:2.20.0`,
    then open <http://localhost:16686>.

=== "Langfuse"

    Langfuse takes OTLP over HTTP: use the OTLP setup with
    `OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel` and
    `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64 of public_key:secret_key>`.

=== "Phoenix"

    ```python
    # pip install arize-phoenix-otel
    from phoenix.otel import register

    register()  # sets the global tracer provider, sending to Phoenix
    pyox_agents.instrument()
    ```

=== "Console"

    ```python
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    pyox_agents.instrument()
    ```

Use a `BatchSpanProcessor` for anything that sends spans over the network: spans are ended on a worker thread,
but a slow exporter behind a `SimpleSpanProcessor` still slows every tool call.

## Your own spans

Spans you open in tools, hooks, approval functions, `on_request` handlers and history stores nest under the
span they ran in, sync or async, and so do spans from auto-instrumented libraries (httpx, SQLAlchemy...).
Nothing to do:

```python
from opentelemetry import trace

tracer = trace.get_tracer("my_app")


@tool
async def search_orders(ctx: RunContext[Db], query: str) -> str:
    """Search the user's orders."""
    with tracer.start_as_current_span("db.search_orders"):  # a child of `execute_tool search_orders`
        return await ctx.deps.search(query)
```

Hooks that run for the whole run (`before_run`, `before_model_request`...) run inside `invoke_agent`; a tool
call's hooks and approval run inside its `execute_tool`.

Native Rust tools and Rust hooks get their `execute_tool` span too, but cannot open spans of their own.

## Subagents, parallel tools and state

- A subagent's `invoke_agent` goes under the `execute_tool` of the call that runs it, in the same trace, with
  `pyox.subagent_id` (the call id, as in its events). Its spans use the subagent's own model.
- Tool calls that run at the same time are sibling spans, each with its own children.
- A state change (`StateChanged`) is a `pyox.state_changed` event on the span of the tool that made it, with the
  changed paths (`pyox.state.paths`, like `["/todos/0"]`). The patch itself only with content capture on.

## Human in the loop

- **Live replies** (`on_request`): the wait is inside the tool call's span, between a `pyox.input_requested` and
  a `pyox.input_replied` event (`pyox.reply.source="on_request"`).
- **Stop and resume** (`answers=`): the waiting `execute_tool` ends with `pyox.tool.waiting=true` (not an error),
  and `invoke_agent` with `pyox.run.pending` (how many requests wait). The resumed run is a **new trace**, under
  whatever is current when you call `stream()` again (usually the new HTTP request). Its call has the same
  `gen_ai.tool.call.id` and a `pyox.input_replied` event (`pyox.reply.source="history"`); with a
  `conversation_id`, both runs have the same `gen_ai.conversation.id`. (A span link from the resumed call to the
  waiting one is planned.)

## Web streams and runs read elsewhere

Spans come from the agent loop, not from whoever reads the events: `async for`, `run.batches()`, `run.ai_sdk()`
and `run.ag_ui(...)` give the same spans with the same times. The run's parent is the span current when
`stream()` is called, so `StreamingResponse(run.ai_sdk())` still goes under the request's span, even though the
stream is read after your handler returned.

`run.stop()` ends the answer being streamed with finish reason `stopped`. If you stop reading instead (`break`,
or a client that disconnects), the run is dropped and its open spans end with `error.type="cancelled"`.

## Content capture

Prompts, answers, system instructions, tool definitions, tool arguments and results, and state patches often hold
personal data, so they are **not recorded by default**. Turn them on with `Instrumentation(capture_content=True)`
(or `instrument(capture_content=True)`), or with the standard environment variable
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY` (`SPAN_AND_EVENT` and `EVENT_ONLY` do the same:
content goes on spans). `capture_content=False` wins over the variable.

`gen_ai.input.messages` holds the whole conversation on every request, so spans grow with it. Limit attribute
sizes with the SDK's `OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT`. Images and files are named (`{"type": "image"}`) but
never copied.

## Metrics

With a meter provider (the global one, or `Instrumentation(meter_provider=...)`), each model request records:

| Metric | Unit | What |
|---|---|---|
| `gen_ai.client.token.usage` | `{token}` | input and output tokens (`gen_ai.token.type`) |
| `gen_ai.client.operation.duration` | `s` | the request, from sending to the last chunk (`error.type` when it failed) |
| `gen_ai.client.operation.time_to_first_chunk` | `s` | from sending to the first streamed chunk |

Each has `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `server.address` and
`server.port`. Run and tool durations are on the spans.

## Spans and attributes

The names follow the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
(as of November 2026). Those conventions are still in development, so **these names are experimental**: they may
change in a minor release of pyox-agents, always with a line in the changelog. Our own attributes start with
`pyox.`.

| Span | Kind | Attributes |
|---|---|---|
| `invoke_agent {name}` | internal | `gen_ai.operation.name`, `gen_ai.provider.name` (and the older `gen_ai.system`), `gen_ai.agent.name`, `gen_ai.request.model`, `gen_ai.conversation.id`, `pyox.subagent_id`, `pyox.run.turns`, `pyox.run.stop_reason`, `pyox.run.pending`, `pyox.usage.*` (the run's tokens, subagents included), `error.type` |
| `chat {model}` | client | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.system`, `gen_ai.request.model`, `gen_ai.request.*` (temperature, top_p, max_tokens, ... as sent, after hooks), `gen_ai.response.id` (Responses API), `gen_ai.response.finish_reasons` (the turn's stop reason: `end`, `length`, `refusal` or `stopped`), `gen_ai.response.time_to_first_chunk`, `gen_ai.usage.*`, `server.address`, `server.port`, `pyox.turn`, `pyox.http.retries`, `pyox.openai.request_id` (for OpenAI support), `error.type`; with content: `gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.tool.definitions` |
| `execute_tool {tool}` | internal | `gen_ai.operation.name`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type`, `gen_ai.tool.description`, `gen_ai.agent.name`, `pyox.tool.kind` (`python`, `rust`, `frontend`, `subagent`, `ask_user`, `ask_parent`), `pyox.tool.waiting`, `error.type` (`tool_error` when the model got an error result); with content: `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`. Events: `pyox.input_requested`, `pyox.input_replied`, `pyox.state_changed` |

Tokens are on `chat` spans only (`gen_ai.usage.*`); runs use `pyox.usage.*`, so backends that add up tokens over
a trace never count them twice. A failed span has status `ERROR` and `error.type` (the exception class, like
`RateLimitError`). A tool error only fails its own span: the run goes on.

## Performance

Off, tracing costs nothing. On, each run, tool call and model request is one or two short Python calls
(on a worker thread, like Python tools), next to model requests of hundreds of milliseconds. Prompts and answers
are only converted for spans when content capture is on and the run is sampled.

## Troubleshooting

- **Nothing appears.** `instrument()` needs an SDK provider: with only `opentelemetry-api`, or before your app sets
  its provider, runs are not traced. The provider is looked up at each run, so setting it later works for later
  runs.
- **Spans are missing at exit.** Batch processors send spans in the background: call `provider.shutdown()` (or
  `provider.force_flush()`) before a script ends.
- **A span processor raises.** Telemetry never breaks a run: the error goes to `sys.unraisablehook` (Python prints
  it) and the run goes on.
