"""Serve an agent to a Vercel AI SDK frontend (`useChat`, assistant-ui).

    from pyox_agents import ai_sdk

    @app.post("/api/chat")
    async def chat(request: Request) -> StreamingResponse:
        stream = ai_sdk.stream(agent, await request.body(), deps=database)
        return StreamingResponse(stream, headers=ai_sdk.HEADERS)

`stream` is `parse` + `agent.execute` + `run.ai_sdk()`. Use them yourself to
check who is asking, or to keep the conversation on the server:

    chat = ai_sdk.parse(await request.body())
    run = agent.execute(chat.prompt, conversation_id=f"{user.id}:{chat.chat_id}", answers=chat.answers)
    return StreamingResponse(run.ai_sdk(), headers=ai_sdk.HEADERS)

The stream itself is made in Rust (`run.ai_sdk()`); this module only reads
requests. Tool calls that wait for a person work with the AI SDK's own
tools: approvals (`addToolApprovalResponse`), and frontend tools and
`ctx.ask` questions (`addToolOutput`; the stream's `data-pyox-request` part
says which calls wait, with the question's `data` and `metadata`).
Approvals need AI SDK 6 or newer.

Shared state: the stream sends the state in a `data-state` part after each
change. It stays in the message, so it comes back with the next request:
`parse` reads the newest one as `ChatRequest.state`, and `stream` runs the
agent with it.

Subagents (`agent.as_tool(...)`) are shown in a `data-subagent` part (id: the
tool call's id) with their status and their own text and tool parts. When a
subagent's tool waits for the user, the subagent's tool call waits: answer it
like any other waiting tool call.

With the history in the browser (the default), a tool that asks more than
once, uses `ctx.step`, or runs a subagent that waits for the user, loses its
earlier answers: keep the conversation on the server (`conversation_id=`) for those.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pyox_agents._core import ProtocolStream
from pyox_agents.agent import Agent
from pyox_agents.human_input import Answer, Approve, Deny, Reply
from pyox_agents.messages import ContentPart, Message, Prompt, Request, RequestKind, ToolCall, ToolMessage

HEADERS = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    "x-vercel-ai-ui-message-stream": "v1",
    "x-accel-buffering": "no",  # stops proxies like nginx from holding the stream back
}
"""Response headers for `run.ai_sdk()`."""


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """What a `useChat` request asks for, ready for `agent.execute(...)`."""

    chat_id: str
    prompt: Prompt | None  # the new user message, if the request has one
    history: list[Message]  # the conversation before it, as the browser has it
    answers: dict[str, Reply]  # replies to tool calls the last run left waiting
    frontend_tools: list[dict[str, Any]] = field(default_factory=list)  # tools the frontend sent, if any
    state: Any = None  # the newest `data-state` part: the state the last run left
    body: dict[str, Any] = field(default_factory=dict)  # the whole request, for your own fields


def stream(agent: Agent, body: bytes | str | Mapping[str, Any], **execute_options: Any) -> ProtocolStream:
    """Runs the agent for a `useChat` request and returns its AI SDK stream.

    `execute_options` go to `agent.execute` (`deps=...`). With `conversation_id=`, the
    history comes from the agent's history store instead of the request. The run's
    state is the one in the request, unless `state=` is given.
    """
    chat = parse(body)
    history = None if "conversation_id" in execute_options else chat.history or None
    execute_options.setdefault("state", chat.state)
    run = agent.execute(
        chat.prompt,
        history=history,
        answers=chat.answers,
        frontend_tools=chat.frontend_tools,
        **execute_options,
    )
    return run.ai_sdk()


def parse(body: bytes | str | Mapping[str, Any]) -> ChatRequest:
    """Reads a `useChat` request body: `{id, messages: [UIMessage, ...], ...}`."""
    request = dict(body) if isinstance(body, Mapping) else json.loads(body)
    ui_messages = list(request.get("messages") or [])

    prompt = None
    if ui_messages and ui_messages[-1]["role"] == "user":
        prompt = _user_content(ui_messages.pop()["parts"])

    history: list[Message] = []
    answers: dict[str, Reply] = {}
    for position, ui_message in enumerate(ui_messages):
        if ui_message["role"] == "user":
            history.append({"role": "user", "content": _user_content(ui_message["parts"])})
        elif ui_message["role"] == "assistant":
            # Answers only count in the last message, when no new prompt replaces them.
            takes_answers = position == len(ui_messages) - 1 and prompt is None
            history.extend(_assistant_messages(ui_message["parts"], answers if takes_answers else None))
        # System messages are left out: the agent's own prompt is the system prompt.

    return ChatRequest(
        chat_id=request.get("id", ""),
        prompt=prompt,
        history=history,
        answers=answers,
        frontend_tools=_frontend_tools(request.get("tools")),
        state=_newest_state(ui_messages),
        body=request,
    )


def _newest_state(ui_messages: list[dict[str, Any]]) -> Any:
    states = [part["data"] for message in ui_messages for part in message["parts"] if part["type"] == "data-state"]
    return states[-1] if states else None


def _user_content(parts: list[dict[str, Any]]) -> str | list[ContentPart]:
    content: list[ContentPart] = []
    for part in parts:
        if part["type"] == "text":
            content.append({"type": "text", "text": part["text"]})
        elif part["type"] == "file":
            url = _checked_url(part["url"])
            if part.get("mediaType", "").startswith("image/"):
                content.append({"type": "image", "url": url})
            else:
                content.append({"type": "file", "url": url, "filename": part.get("filename") or "file"})
    if len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]
    return content


def _checked_url(url: str) -> str:
    # Never let a browser make the server read a local path or an internal address.
    if not url.startswith(("data:", "https://")):
        raise ValueError(f"Files must be data: or https:// URLs, got {url[:40]!r}")
    return url


def _assistant_messages(parts: list[dict[str, Any]], answers: dict[str, Reply] | None) -> list[Message]:
    """One assistant message per step (model turn), each followed by its tool results.

    With `answers`, replies the user gave in this message are added to it (by the id of
    the request: the call id, or `call_0/call_1` for a subagent's call), and their calls
    get a *waiting* tool message so the run can continue them.
    """
    waiting_requests = [
        part["data"] for part in parts if part["type"] == "data-pyox-request" and part["data"].get("waiting")
    ]
    waiting_kinds: dict[str, RequestKind] = {request["toolCallId"]: request["kind"] for request in waiting_requests}
    request_ids = {
        request["toolCallId"]: request.get("requestId", request["toolCallId"]) for request in waiting_requests
    }
    messages: list[Message] = []
    for step in _split_steps(parts):
        text = "".join(part["text"] for part in step if part["type"] == "text")
        tool_parts = [part for part in step if _is_tool_part(part)]
        if not text and not tool_parts:
            continue
        tool_calls: list[ToolCall] = [
            {"id": part["toolCallId"], "name": _tool_name(part), "arguments": part.get("input") or {}}
            for part in tool_parts
        ]
        messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})
        for part in tool_parts:
            reply = _reply(part, waiting_kinds) if answers is not None else None
            if answers is not None and reply is not None:
                answers[request_ids.get(part["toolCallId"], part["toolCallId"])] = reply
            messages.append(_tool_message(part, waiting_kinds, reply))
    return messages


def _split_steps(parts: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    steps: list[list[dict[str, Any]]] = [[]]
    for part in parts:
        if part["type"] == "step-start":
            steps.append([])
        else:
            steps[-1].append(part)
    return steps


def _is_tool_part(part: dict[str, Any]) -> bool:
    return part["type"].startswith("tool-") or part["type"] == "dynamic-tool"


def _tool_name(part: dict[str, Any]) -> str:
    return part["toolName"] if part["type"] == "dynamic-tool" else part["type"].removeprefix("tool-")


def _reply(part: dict[str, Any], waiting_kinds: dict[str, RequestKind]) -> Reply | None:
    """The user's reply in this tool part, if it answers a waiting call."""
    if part["state"] == "approval-responded":
        approval = part["approval"]
        if approval.get("approved"):
            return Approve()
        return Deny(approval.get("reason") or "The user denied this tool call")
    if part["toolCallId"] not in waiting_kinds:
        return None  # an output the server sent: a finished call
    if part["state"] == "output-available":
        return Answer(part.get("output"))
    if part["state"] == "output-error":
        return Deny(part.get("errorText") or "The tool failed")
    return None


def _tool_message(part: dict[str, Any], waiting_kinds: dict[str, RequestKind], reply: Reply | None) -> ToolMessage:
    call_id, name = part["toolCallId"], _tool_name(part)
    state = part["state"]
    if reply is None and state == "output-available" and call_id not in waiting_kinds:
        output = part.get("output")
        content = output if isinstance(output, str) else json.dumps(output)
        return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content, "is_error": False}
    if reply is None and state in ("output-error", "output-denied"):
        error = part.get("errorText") or "The user denied this tool call"
        return {"role": "tool", "tool_call_id": call_id, "name": name, "content": error, "is_error": True}
    # No result yet: a waiting call, answered by `reply` or not.
    kind: RequestKind = "approval" if state.startswith("approval") else waiting_kinds.get(call_id, "input")
    request: Request = {"kind": kind, "data": None, "metadata": None, "answer": None}
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": "", "requests": [request]}


def _frontend_tools(tools: Any) -> list[dict[str, Any]]:
    """Tool definitions some frontends send (assistant-ui): `{name: {description, parameters}}` or a list."""
    if isinstance(tools, Mapping):
        return [{"name": name, **definition} for name, definition in tools.items()]
    return list(tools or [])
