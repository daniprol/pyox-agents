# Streaming

`agent.execute(prompt)` starts a run and returns it. Iterate over it with `async for` to get its events as they
happen:

```python
--8<-- "examples/basic.py"
```

## Events

Events are small, read-only objects backed by Rust. They work with `isinstance` and with `match`.

| Event          | When                       | Fields                                                     |
|----------------|----------------------------|------------------------------------------------------------|
| `TextDelta`    | the model streams text     | `text`                                                     |
| `ToolStarted`  | a tool call starts         | `call_id`, `name`, `arguments` (dict)                      |
| `ToolFinished` | a tool call ends           | `call_id`, `name`, `result`, `is_error`, `metadata`        |
| `InputRequested` | a tool call waits for a person ([human in the loop](human-in-the-loop.md)) | `call_id`, `kind`, `data`, ... |
| `StateChanged` | a tool changed the [state](state.md) | `state`, `delta`                                 |
| `SubagentStarted`, `SubagentFinished` | a [subagent](subagents.md) starts or stops | `call_id`, `name`, ...      |
| `RunFinished`  | the run ends (always last) | `text`, `turns`, `usage`, `response_id`, `messages`, `new_messages`, `pending`, `state` |

A subagent's events come in the same stream, with `subagent_id` set (it is `None` for the main agent's events).

`await agent.run(prompt)` does the same run without streaming and returns the `RunFinished` event.

One `Agent` can serve many runs at the same time: each run has its own history. Create it once and reuse it.

## Token usage

`RunFinished.usage` holds the tokens used by all the model requests of the run, with both APIs:

| Field                 | Meaning                                                           |
|-----------------------|-------------------------------------------------------------------|
| `input_tokens`        | tokens sent to the model                                          |
| `cached_input_tokens` | the part of `input_tokens` read from the prompt cache (cheaper)   |
| `output_tokens`       | tokens the model generated                                        |
| `reasoning_tokens`    | the part of `output_tokens` spent reasoning (reasoning models)    |
| `requests`            | model requests that reported usage                                |

Counts the API did not report are 0. If `requests` is lower than `turns`, some requests sent no usage (some
OpenAI-compatible servers don't).

```python
result = await agent.run("Hi")
print(result.usage.input_tokens, result.usage.output_tokens)

total = Usage()  # all zeros
for result in results:
    total += result.usage
```

Tools and hooks see the tokens used so far as `ctx.usage`. A failed run still used (and paid for) tokens: an
`AgentError` raised by a run has them in `error.usage`.

```python
try:
    result = await agent.run("Hi")
except AgentError as error:
    print("used before failing:", error.usage)
```

## Grouping streamed text

Models stream text in tiny pieces, and each `TextDelta` costs a little Python work. To get fewer, bigger events,
choose how text is grouped with `text_chunks`:

| `text_chunks`        | One `TextDelta` per...                                             |
|----------------------|--------------------------------------------------------------------|
| `"delta"` (default)  | piece sent by the API (lowest latency)                             |
| `10` (any number N)  | N pieces                                                           |
| `"paragraph"`        | markdown paragraph; fenced code blocks are never split             |
| `"message"`          | whole assistant message                                            |

Set it on the agent, or override it for one run:

```python
agent = Agent(model="gpt-5-mini", text_chunks="paragraph")

async for event in agent.execute("Explain asyncio", text_chunks=20):
    ...
```

!!! tip "Which one?"
    For a live UI, a number like `10` keeps almost all the savings with no visible delay.
    `"paragraph"` and `"message"` are cheapest but make the user wait for the whole paragraph or answer.
    See the [benchmarks](benchmarks.md#streaming-modes-text_chunks).

## Batches of events

Each step of `async for event in run` hands one event from Rust to Python, and the hand-over costs more than most
events. `run.batches()` hands over every event that is already waiting, as a list:

```python
async for events in run.batches():
    for event in events:
        ...
```

It never waits to fill a batch: when events come one at a time (a real model streaming at its own pace), each
list has one event and nothing changes. When they pile up (cached or very fast answers, a busy server falling
behind), one step carries them all. `run.ai_sdk()` and `run.ag_ui(...)` (see [Web frontends](web-frontends.md))
read the run this way too. Numbers in the [benchmarks](benchmarks.md#serving-a-stream-runbatches-and-runai_sdk).

## Other settings

```python
agent = Agent(
    model="gpt-5-mini",
    prompt="Be concise.",                  # system prompt
    model_parameters={"temperature": 0.2},  # sent as-is to the API
    max_turns=20,                           # model requests per run before MaxTurnsExceededError
)
```
