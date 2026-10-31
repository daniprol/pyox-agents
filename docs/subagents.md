# Subagents

A subagent is an agent another agent calls like a tool. Give it its own model, prompt and tools, and pass it to
the main agent with `as_tool`:

```python
from pyox_agents import Agent

researcher = Agent(model="gpt-5-mini", prompt="Research topics. Be thorough.", tools=[web_search])
writer = Agent(
    model="gpt-5",
    prompt="Write articles. Use `research` to find facts.",
    tools=[researcher.as_tool("research", "Research a topic and report what you found.")],
)

result = await writer.run("Write a short article about Vigo")
```

The model calls `research(task="...")`. The researcher starts a fresh conversation with the task as its first
message, works with its own tools, and its final answer is the tool's result. It knows nothing else about the
main conversation, so the main agent must put everything it needs in the task.

## What a subagent shares

| Shared with the main agent               | Its own                                     |
|------------------------------------------|---------------------------------------------|
| `deps` (its tools get the same `ctx.deps`) | model, prompt, tools, hooks, `max_turns`  |
| the person: approvals and `ctx.ask` questions | its conversation                       |
| the run's events and token usage         |                                             |
| the run's [state](state.md), if allowed  |                                             |

A subagent cannot have subagents of its own: one level keeps runs easy to follow.

If the subagent fails (the API is down, too many turns), the main agent gets the error as the tool's result and
can react to it.

## Options

```python
researcher.as_tool(
    "research",
    "Research a topic.",
    inherit_tools=False,  # True: also give it the main agent's tools
    talk_to=None,         # None, "user" or "parent": who it can ask questions
    state=None,           # None, "read" or "write": what it may do with the run's state
)
```

**Tools.** By default a subagent has only its own tools. With `inherit_tools=True` it also gets the main agent's
tools (never its subagents). When names clash, its own tool wins. Tools keep their approval rules wherever they run.

**Talking to the user.** With `talk_to="user"`, the subagent gets an `ask_user(question)` tool. The question
waits for the person like a [`ctx.ask`](human-in-the-loop.md), and the answer is the tool's result.

**Talking to the main agent.** With `talk_to="parent"`, the subagent gets an `ask_parent(question)` tool. When it
asks, it stops, and the main agent reads the question as the tool's result:

```text
The subagent asks: How many days is the trip?

To answer, call plan with subagent_id="call_0" and your answer as the task.
```

The main agent answers by calling the tool again with that `subagent_id`: the subagent goes on with its own
conversation, with the answer as a new message. Its final answers end with the `subagent_id` too, so the main
agent can follow up later in the same way.

**State.** A subagent sees the run's [state](state.md) only if its tool says so: `state="read"` lets its tools
read `ctx.state`, `state="write"` lets them change it too.

## Events

A subagent's events come in the same stream as the main agent's, marked with `subagent_id`: the id of the tool
call running it (`None` for the main agent's events). Its tool call ids start with it (`call_0/call_1`), so ids
stay unique:

```python
async for event in writer.execute("Write about Vigo"):
    match event:
        case SubagentStarted(name=name, task=task):
            print(f"[{name}] {task}")
        case TextDelta(text=text, subagent_id=None):
            print(text, end="")  # only the main agent's text
        case SubagentFinished(name=name, usage=usage):
            print(f"[{name}] done ({usage.output_tokens} tokens)")
```

| Event              | Fields                                                                 |
|--------------------|------------------------------------------------------------------------|
| `SubagentStarted`  | `call_id` (its id), `name` (the tool's), `task`                        |
| `SubagentFinished` | `call_id`, `name`, `result`, `is_error`, `waiting`, `usage` (its tokens, also in the run's) |

[Web frontends](web-frontends.md#shared-state-and-subagents) get them as AG-UI `SUBAGENT_*` events, or an AI SDK
`data-subagent` part.

## Approvals and questions

A subagent's approvals, `ctx.ask` questions and frontend tools reach the person exactly like the main agent's
([human in the loop](human-in-the-loop.md)). Their ids say where they come from: `call_0/call_1` is the
subagent's call `call_1`, inside the main agent's call `call_0`, and `InputRequested.subagent_id` is `"call_0"`.

- **Live**: `on_request` answers them while the subagent waits.
- **Stop and resume**: the run stops, and `RunFinished.pending` lists them. The main agent's call `call_0` waits
  too, and keeps the subagent's conversation in its tool message (under `subagent`, never sent to the model).
  Answer with the same ids:

```python
stopped = await assistant.run("Book me a flight to Vigo")
[request] = stopped.pending  # call_id="call_0/call_0", tool_name="book", kind="approval"
result = await assistant.run(history=stopped.messages, answers={"call_0/call_0": Approve()})
```

The subagent goes on from where it stopped, then the main agent does.
