# Sync code

Every way to run an agent has a sync twin, for code without an event loop: scripts, Django and Flask views, Celery
tasks, notebooks.

| Async                             | Sync                            |
|-----------------------------------|---------------------------------|
| `await agent.run(...)`            | `agent.run_sync(...)`           |
| `async for event in agent.stream(...)` | `for event in agent.stream_sync(...)` |
| `async for events in run.batches()` | `for events in run.batches()` |
| `async for text in run.ai_sdk()`, `run.ag_ui(...)` | `for text in run.ai_sdk()`, `run.ag_ui(...)` |
| `await ctx.ask(...)`, `await ctx.step(...)` | `ctx.ask_sync(...)`, `ctx.step_sync(...)` |

The arguments are the same.

```python
--8 < --"examples/sync.py"
```

## How it works

The agent loop already runs in Rust, so a sync run needs no event loop at all. Your thread waits in Rust with the
GIL released (other threads keep running), and runs your sync Python code when the run needs it:

| Python code | async run | sync run |
|---|---|---|
| sync tools, hooks, history stores, `on_request` | a worker thread | **your thread** |
| `@tool(run_in_thread=True)` | a worker thread | a worker thread |
| async tools, hooks, stores, handlers | your event loop | pyox's own event loop thread |
| Rust tools | a worker thread, no Python | the same |

So sync functions never block an event loop, and in sync runs they behave like any function you call: Django's ORM
sees the view's transaction, and thread-locals and context variables just work. Parallel tool calls of a sync run
run one after the other; give slow tools `run_in_thread=True` to run them in parallel.

Async tools and MCP servers work in sync runs too. They run on one event loop thread that pyox starts the first time
it needs it, and keeps for the life of the process, so async clients made by one run work in the next.

## Django and Flask

```python
from django.http import JsonResponse, StreamingHttpResponse
from pyox_agents import ai_sdk


def chat(request):
    result = agent.run_sync(request.POST["prompt"], deps=request.user)
    return JsonResponse({"text": result.text})


def chat_stream(request):  # for `useChat`, see Web frontends
    chat = ai_sdk.parse(request.body)
    run = agent.stream_sync(
        chat.prompt, history=chat.history, answers=chat.answers, frontend_tools=chat.frontend_tools, state=chat.state
    )
    return StreamingHttpResponse(run.ai_sdk(), headers=ai_sdk.HEADERS)
```

When the client goes away, the server closes the stream, which ends the run.

A sync run holds its thread until the model answers (seconds), like any sync view that calls an API. Use Gunicorn's
`gthread` workers to serve several runs per process, or the async API with an ASGI server for many at once.

## Stopping

- `break`, `run.close()` or leaving `with agent.stream_sync(...) as run:` ends the run at once.
- `run.stop()` ends it at the next safe point and keeps what it did, from any thread:
  `threading.Timer(60, run.stop).start()` gives a run at most a minute.
- Ctrl+C raises `KeyboardInterrupt` within 0.1 s and ends the run, also inside a tool.

## Limits

- **gevent and eventlet workers:** a sync run blocks the worker's other greenlets until it ends. Use `gthread`
  workers, or the async API.
- **Forking:** a process forked after pyox-agents ran (Gunicorn `--preload` with a run in the master, Celery
  prefork with a run at import) cannot run agents and raises a `RuntimeError` that says so. Run agents only after
  the fork.
- **Inside async code:** `run_sync` works inside a running event loop (a notebook cell), but blocks it for the whole
  run. In `async def` functions use `await agent.run(...)`.
- **MCP servers:** open them with `with McpTools...` (not `async with`) for sync runs.
