# Errors

There are two kinds of failures, and they go to different places:

- **A tool fails**: the model reads what went wrong and decides what to do next. The run goes on.
- **Anything else fails** (the API, the network, a hook, a history store): the run raises an `AgentError`.

## When a tool fails

Raise `ToolFailed` to tell the model exactly what happened. Write the message for the model: what went wrong and
what it could try instead.

```python
from pyox_agents import ToolFailed, tool


@tool
def search_flights(origin: str, date: str) -> list[str]:
    """Search flights."""
    flights = find_flights(origin, date)
    if not flights:
        raise ToolFailed(f"No flights from {origin} on {date}. Try another date.")
    return flights
```

Any other exception is a bug or something you did not expect. The model reads it with its type, like
`FileNotFoundError: No such file: 'app.yml'`, and pyox logs it as a warning (with the traceback) on the
`pyox_agents` logger, so it does not go unnoticed.

Either way, `ToolFinished.is_error` is `True` and the run continues.

## Choosing what the model reads

The default text may say too much (a file path, a SQL query) or too little. `on_tool_error` decides, for every
tool of an agent. It gets the exception itself, and returns the text for the model, or `None` to keep the
default:

```python
from pyox_agents import Agent, Hooks, RunContext, ToolCall


def friendly_errors(ctx: RunContext, call: ToolCall, error: Exception) -> str | None:
    if isinstance(error, PermissionError):
        return "You are not allowed to do that. Tell the user."
    if isinstance(error, ConnectionError):
        return "The service is down. Try again later."
    return None  # the default text


agent = Agent(model="gpt-5-mini", tools=[...], hooks=[Hooks(on_tool_error=friendly_errors)])
```

A single tool can have its own, which is asked first:

```python
def hide_paths(ctx: RunContext, call: ToolCall, error: Exception) -> str | None:
    return "The file could not be read." if isinstance(error, OSError) else None


@tool(on_error=hide_paths)
def read_report(name: str) -> str:
    """Read a report."""
    return open(f"/srv/reports/{name}.txt").read()
```

`on_tool_error` is called for every failed call: exceptions from the tool, [bad arguments](#bad-arguments), an
unknown tool name, a [timeout](#timeouts). It may be async. If it raises, the run ends with a `HookError`.

## Ending the run

Some failures should stop everything: expired credentials, a payment service that is down. Raise `StopRun` in a
tool or in any hook:

```python
from pyox_agents import StopRun


def stop_on_auth_errors(ctx: RunContext, call: ToolCall, error: Exception) -> None:
    if isinstance(error, PermissionError):
        raise StopRun("The credentials expired") from error
```

The run raises it. Like every `AgentError`, it has the conversation so far, which you can continue later:

```python
try:
    result = await agent.run("Pay my bills")
except StopRun as error:
    print(f"Stopped: {error}")
    await save(error.messages)  # then `agent.run("Try again", history=error.messages)`
```

## Bad arguments

Models sometimes call a tool with arguments that do not fit: a missing field, a string for a number, a value
out of range. pyox checks the arguments against the tool's type hints (and pydantic validates its models) before
calling it. The model reads one short line per problem:

```text
Invalid arguments for book_trip:
- trip.days: Input should be less than or equal to 7 (got 10)
- class_: expected one of "economy", "business", got "first"
- seats: missing
```

The model then decides whether to call the tool again. pyox never retries for it: no hidden loops, and every
attempt is a normal tool call you can see in the stream and in traces. `max_turns` (20 by default) is the limit
for a model that keeps trying.

To stop sooner, count failures in `on_tool_error`. The exception is an `InvalidArguments` (a `ToolFailed`) with
the problems in `error.errors`:

```python
from collections import Counter

from pyox_agents import InvalidArguments


def two_tries(ctx: RunContext[Counter[str]], call: ToolCall, error: Exception) -> str | None:
    if isinstance(error, InvalidArguments):
        ctx.deps[call["name"]] += 1
        if ctx.deps[call["name"]] >= 2:
            return f"{error}\nDo not call {call['name']} again. Tell the user what is missing."
    return None


await agent.run("Book me a trip", deps=Counter())
```

## Timeouts

A tool that may hang (a slow website, a stuck query) can have a time limit:

```python
@tool(timeout_seconds=30)
async def fetch_page(url: str) -> str:
    """Fetch a web page."""
    ...
```

After 30 seconds the model reads `TimeoutError: Tool "fetch_page" timed out after 30 seconds`, and an async tool
is cancelled. A sync tool cannot be interrupted: it finishes in the background and its result is dropped.

## Errors the run raises

Everything pyox raises is an `AgentError`, so one `except` catches them all:

| Exception               | Raised when                                                        |
|-------------------------|--------------------------------------------------------------------|
| `ApiError`              | the API answered with an error                                     |
| `AuthenticationError`   | the API key is missing or wrong (an `ApiError`)                    |
| `RateLimitError`        | the API is rate limiting you, after the retries (an `ApiError`)    |
| `RequestTimeoutError`   | a request took longer than its timeout                             |
| `NetworkError`          | the server could not be reached                                    |
| `InvalidResponseError`  | the API sent something that could not be parsed                    |
| `MaxTurnsExceededError` | the run made more than `max_turns` model requests                  |
| `HookError`             | a hook or `on_request` raised, or returned something invalid       |
| `HistoryError`          | a history store failed                                             |
| `StopRun`               | a tool or hook [ended the run](#ending-the-run)                    |

When your own code raised (a hook, a history store), its exception is the `__cause__`, so the traceback shows
both. Every `AgentError` raised by a run has `messages` (the conversation so far) and `usage` (the tokens spent).

```python
from pyox_agents import AgentError, ApiError, RateLimitError

try:
    result = await agent.run("Hi")
except RateLimitError as error:
    print(f"Slow down, try again in {error.retry_after_seconds} seconds")
except ApiError as error:
    print(error.status_code, error.code, error.request_id)
except AgentError as error:
    print(f"The run failed: {error}")
```

An `ApiError` has the details as attributes, so there is no message to parse:

| Attribute             | What it holds                                                          |
|-----------------------|------------------------------------------------------------------------|
| `status_code`         | the HTTP status (`None` for an error that came inside a stream)        |
| `code`                | OpenAI's error code, like `"context_length_exceeded"`, if it gave one  |
| `body`                | the error body, as parsed JSON when it is JSON (`{"error": {...}}`)    |
| `request_id`          | OpenAI's id for the request, for support                               |
| `retry_after_seconds` | how long the server asked to wait, if it said                          |

Rate limits and server errors are retried before they raise (see [Client: retries](client.md#retries)).

A few things are not `AgentError`s, on purpose:

- **Mistakes in your setup** raise right away, when you create the agent or the client: `TypeError` or
  `ValueError`, like `@tool(timeout_seconds=0)`.
- **Cancelling** the task that reads the run (`asyncio.CancelledError`) or Ctrl+C (`KeyboardInterrupt`) stops the
  run and its tools, and goes up as usual.
