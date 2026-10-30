# Web frontends (AI SDK and AG-UI)

pyox-agents speaks the stream formats of two frontend toolkits, so a chat UI built with them can talk to your
agent directly:

| Frontend                                                   | Python module          | Stream            |
|------------------------------------------------------------|------------------------|-------------------|
| [Vercel AI SDK](https://ai-sdk.dev) 6 and 7 (`useChat`, assistant-ui) | `pyox_agents.ai_sdk` | `run.ai_sdk()` |
| [AG-UI](https://docs.ag-ui.com) 1.0 (CopilotKit, `HttpAgent`) | `pyox_agents.ag_ui`  | `run.ag_ui(thread_id, run_id)` |

The streams are made in Rust, straight from the run's events: Python only passes the finished text to your web
server. The modules read the requests in Python, where they are easy to change.

## A server for both

```python
--8<-- "examples/web_server.py"
```

`stream(agent, body, **options)` reads the request, runs the agent and returns the stream. `options` go to
`agent.execute` (`deps=...`, `text_chunks=...`).

## The pieces, for more control

`stream` is three steps. Do them yourself to check who is asking, read your own fields from the request, or keep
the conversation on the server:

```python
@app.post("/api/chat")
async def chat(request: Request, user: User = Depends(current_user)) -> StreamingResponse:
    chat = ai_sdk.parse(await request.body())  # chat.prompt, chat.history, chat.answers, chat.body
    run = agent.execute(chat.prompt, conversation_id=f"{user.id}:{chat.chat_id}", answers=chat.answers, deps=user)
    return StreamingResponse(run.ai_sdk(), headers=ai_sdk.HEADERS)
```

`ag_ui.parse` returns the same fields (`prompt`, `history`, `answers`, `frontend_tools`), plus `thread_id`,
`run_id`, `state`, `context` and `forwarded_props`.

**Where the conversation lives.** By default it comes from the browser with each request (`history`). With
`conversation_id=`, the agent's [history store](history.md) keeps it instead and the browser's copy is ignored:
only the new prompt and the answers are used. Keep it on the server when you can't trust the browser with it, and
for tools that ask more than once or use `ctx.step` (their earlier answers are not in the browser's copy).

## Tools that wait for a person

[Approvals, questions and frontend tools](human-in-the-loop.md) use each toolkit's own way of waiting:

=== "AI SDK"

    | Waiting call         | In the stream                                     | The browser replies with                   |
    |----------------------|---------------------------------------------------|--------------------------------------------|
    | approval             | `tool-approval-request` (approval id = call id) and a `data-pyox-request` part | `addToolApprovalResponse({id, approved, reason})` |
    | `ctx.ask` question   | a `data-pyox-request` part                         | `addToolOutput({toolCallId, output})`      |
    | frontend tool        | a `data-pyox-request` part                         | `addToolOutput({toolCallId, output})` (or `onToolCall`) |

    Every waiting call gets a `data-pyox-request` part. It has the call's id as its id, and
    `{toolCallId, toolName, waiting: true, kind, key, data, metadata}` as its data: `data` is what the tool asked,
    `metadata` what it left for your UI. When the call goes on, the part is replaced with `waiting: false`.
    Send the reply with `sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses` (or your own
    check), and the next request carries it.

=== "AG-UI"

    | Waiting call         | In the stream                                                          | The client replies with |
    |----------------------|------------------------------------------------------------------------|-------------------------|
    | approval             | `RUN_FINISHED` outcome `interrupt`, reason `approval`, with a `responseSchema` | `resume` payload `{approved, editedArgs?, reason?}` |
    | `ctx.ask` question   | `RUN_FINISHED` outcome `interrupt`, reason `input`, `message` = the question | `resume` payload: the answer |
    | frontend tool (the request's `tools`) | `RUN_FINISHED` outcome `success`, with `pendingToolCallIds` | a tool message with the result |

    Each interrupt's id is the tool call's id. A `cancelled` resume denies the call.

Results of frontend tools are not sent back to the frontend: it made them.

## Errors

A failed run (the API is down, too many turns) ends the stream with the toolkit's error: an `error` chunk for the
AI SDK, `RUN_ERROR` (with the error's class name as `code`) for AG-UI. Other exceptions (bugs) are raised as usual.

## Not supported (yet)

- AG-UI shared state: `state` is passed to your code on `RunInput`, but the agent does not read or send state
  updates. Subagents and reasoning events are not sent either.
- AI SDK 5 and older: use AI SDK 6 or newer.
- File parts must be `data:` or `https://` URLs, so a browser can't make your server read a local path.
