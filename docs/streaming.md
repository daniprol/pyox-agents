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
| `ToolFinished` | a tool call ends           | `call_id`, `name`, `result`, `is_error`                    |
| `RunFinished`  | the run ends (always last) | `text`, `turns`, `input_tokens`, `output_tokens`, `cached_input_tokens`, `response_id`, `messages`, `new_messages` |

`await agent.run(prompt)` does the same run without streaming and returns the `RunFinished` event.

One `Agent` can serve many runs at the same time: each run has its own history. Create it once and reuse it.

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

## Other settings

```python
agent = Agent(
    model="gpt-5-mini",
    prompt="Be concise.",                  # system prompt
    model_parameters={"temperature": 0.2},  # sent as-is to the API
    max_turns=20,                           # model requests per run before MaxTurnsExceededError
)
```
