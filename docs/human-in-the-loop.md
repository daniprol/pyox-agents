# Human in the loop

A tool call can wait for a person: to approve it, to answer a question the tool asks, or to run it in your app
(a frontend tool). No database or checkpoints are needed: everything a waiting call needs is in the history.

## Three ways to wait

**Approval.** The call only runs once someone approves it:

```python
from pyox_agents import RunContext, tool


@tool(requires_approval=True)
def delete_file(path: str) -> str:
    """Delete a file."""
    ...


def outside_tmp(ctx: RunContext, arguments: dict) -> bool:  # or decide per call
    return not arguments["path"].startswith("/tmp/")


@tool(requires_approval=outside_tmp)
def remove(path: str) -> str: ...
```

A [`before_tool_call` hook](hooks.md) can ask for approval too, by raising `ApprovalRequired(data, metadata=...)`.

**A question.** `ctx.ask` waits for an answer anywhere in the tool body:

```python
@tool
async def book_flight(ctx: RunContext, city: str) -> str:
    """Book a flight."""
    options = await find_flights(city)
    answer = await ctx.ask({"options": options}, metadata={"widget": "flight-picker"})
    return f"Booked {answer.value}"
```

**A frontend tool.** Your app runs it (for example in the browser) and sends back its result:

```python
@tool(frontend=True)
def pick_color(options: list[str]) -> str:
    """Let the user pick a color."""


get_location = Tool.frontend_tool("get_location", "Get the user's location.", {"type": "object"})
```

Or give a run the tools your frontend sent with the request, as JSON schemas, for that run only:

```python
run = agent.execute(prompt, frontend_tools=[{"name": "get_location", "description": "...", "parameters": {...}}])
```

Each one becomes an `InputRequested` (with `call_id`, `tool_name`, `arguments`, `kind`: `"approval"`, `"input"`
or `"frontend"`, `key`, `data` and `metadata`). Reply with:

| Reply                                    | For                       | Effect                                                    |
|------------------------------------------|---------------------------|-----------------------------------------------------------|
| `Approve(arguments=None, metadata=None)` | approvals                 | the tool runs (with other `arguments`, if given)          |
| `Answer(value, metadata=None)`           | questions, frontend tools | `ctx.ask` returns it; a frontend tool's result is `value` |
| `Deny(message, metadata=None)`           | all                       | the model gets `message` as an error result               |

## Two ways to reply

**Live.** Pass `on_request`. It replies while the run waits, and the tool goes on from where it asked:

```python
async def ask_the_user(request: InputRequested) -> Reply | None:
    return await websocket.ask(request)  # return None to stop the run instead


result = await agent.run("Fly me to Paris", on_request=ask_the_user)
```

**Stop and resume.** Without a handler, the run stops after the tools, and `RunFinished.pending` lists the
requests (they are also streamed as `InputRequested` events). Send the answers in a later run, with the history
of the stopped one (or its `conversation_id`):

```python
result = await agent.run("Fly me to Paris")
if result.pending:
    # Later, maybe in another HTTP request or process:
    result = await agent.run(history=result.messages, answers={"call_0": Answer("Paris 18:00")})
```

Calls that don't wait finish normally and are not run again. A waiting tool **runs again from its start**: earlier
`ctx.ask` calls return their stored answers at once. So do side effects after the question, or inside
`ctx.step`, which saves a result in the history and returns it on later runs instead of calling the function again:

```python
@tool
async def buy(ctx: RunContext, amount: int) -> str:
    receipt = await ctx.step("charge", charge_card, amount)  # charged once, even if the tool runs again
    answer = await ctx.ask(f"Email receipt {receipt['id']}?")
    ...
```

Step names must be unique within a call, and step results must be JSON. A tool asking several questions matches
answers to questions by order; give them a `key=` (`ctx.ask(..., key="email")`) to match by name.

## Visible data and hidden metadata

`data` (what the tool asks) and the tool's result reach the model as usual. `metadata` never does: it is kept in
the history for your app and frontend. That holds for both directions:

- the request's `metadata` (`ctx.ask(..., metadata=...)`, `ApprovalRequired(..., metadata=...)`);
- the reply's `metadata`, which the tool reads (`answer.metadata`, `ctx.approval.metadata`);
- a result's `metadata`: return `ToolReturn(content, metadata=...)` from any tool; it shows up in
  `ToolFinished.metadata` and the tool message.

## A new prompt instead of answers

If a run gets a new prompt while calls still wait, by default each waiting call gets the result
"The user did not answer" and the run goes on. With `Agent(unanswered="raise")` it raises
`PendingRequestsError` instead. Continuing such a history with neither a prompt nor answers raises `ValueError`.

## The history

A waiting call is a tool message with an empty `content` and a `requests` list (each with `kind`, `key`, `data`,
`metadata` and `answer`, which is `None` while waiting). The model never sees waiting messages. When the call
resumes, a new tool message is added after it, so history only grows and history stores just append.
`after_run` hooks see stopped runs too, with `result.pending` set.

## Subagents

A [subagent](subagents.md)'s approvals and questions work the same way, with ids like `call_0/call_1` (the
subagent's call `call_1`, inside the main agent's call `call_0`): answer them with those ids.

## Example

```python
--8<-- "examples/human_in_the_loop.py"
```
