# State

A run can carry a *state*: one JSON document (usually a dict) that tools read and change, and that your app sees
change as it happens. Think of a todo list, a shopping cart or a document the agent and the user edit together.

```python
from pyox_agents import Agent, RunContext, StateChanged, tool


@tool
def add_todo(ctx: RunContext, text: str) -> str:
    """Add a todo to the list."""
    ctx.state["todos"].append(text)
    return f"Added {text!r}"


agent = Agent(model="gpt-5-mini", tools=[add_todo])

async for event in agent.execute("Plan my trip", state={"todos": []}):
    if isinstance(event, StateChanged):
        print(event.delta)  # [{"op": "add", "path": "/todos/0", "value": "Book the flight"}]
```

- Give it with `execute(state=...)` (or `run(state=...)`). Anything JSON works.
- Inside a tool, `ctx.state` reads it. Change it in place, or replace it: `ctx.state = {...}`.
- When the tool finishes, its changes are applied and sent as a `StateChanged` event, with the whole new `state`
  and a `delta`: what changed, as a [JSON Patch](https://jsonpatch.com/).
- `RunFinished.state` is the state at the end of the run.

Nothing is stored: your app keeps the state between runs and gives it again. Your own dict is never changed.

## Rules

- **Changes apply when a tool finishes.** A tool that fails, or stops to wait for a person, changes nothing: it
  runs again from its start once answered, so it would make its changes twice otherwise.
- **Tools running at the same time keep each other's changes.** Each tool applies only what it changed, so two
  tools changing different keys never overwrite each other.
- **Hooks read a copy.** Only tools change the state.
- **The model does not see the state** unless a tool returns it (or a [hook](hooks.md) adds it to the prompt).

## Subagents

A [subagent](subagents.md) sees the state only if its tool allows it: `as_tool(..., state="read")` or
`state="write"`. Without it, `ctx.state` raises `PermissionError` in the subagent's tools, and a read-only
subagent's tool that changes the state gets an error result instead.

## Web frontends

AG-UI clients (CopilotKit) and AI SDK frontends share the state with the agent: `ag_ui.stream` and
`ai_sdk.stream` start the run with the client's state, and send each change as it happens (a `STATE_DELTA`
event, or a `data-state` part). See [web frontends](web-frontends.md#shared-state-and-subagents).
