"""Serve an agent to an AG-UI client (CopilotKit and others), AG-UI 1.0.

    from pyox_agents import ag_ui

    @app.post("/agent")
    async def run_agent(request: Request) -> StreamingResponse:
        stream = ag_ui.stream(agent, await request.body(), deps=database)
        return StreamingResponse(stream, headers=ag_ui.HEADERS)

`stream` is `parse` + `agent.execute` + `run.ag_ui(thread_id, run_id)`. Use them
yourself to check who is asking, or to keep the conversation on the server:

    run_input = ag_ui.parse(await request.body())
    run = agent.execute(run_input.prompt, conversation_id=run_input.thread_id,
                        answers=run_input.answers, frontend_tools=run_input.frontend_tools)
    return StreamingResponse(run.ag_ui(run_input.thread_id, run_input.run_id), headers=ag_ui.HEADERS)

The stream itself is made in Rust; this module only reads requests.

Waiting tool calls: an approval or a `ctx.ask` question ends the run with an
`interrupt` outcome whose interrupt id is the tool call id. The client
resumes with `resume: [{"interruptId": ..., "status": "resolved", "payload": ...}]`:
`{"approved": true, "editedArgs": {...}}` or `{"approved": false, "reason": "..."}`
for an approval, any value for a question. The client's tools (`tools` in the
request) become frontend tools: the client runs them and sends back tool messages.

`state`, `context` and `forwarded_props` are kept on `RunInput` for your code
(e.g. pass them as `deps`); the agent does not read or change state on its own.
With the history in the client (the default), a tool that asks more than once,
or uses `ctx.step`, loses its earlier answers: keep the conversation on the
server (`conversation_id=`) for those.
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
    "x-accel-buffering": "no",  # stops proxies like nginx from holding the stream back
}
"""Response headers for `run.ag_ui(...)`."""


@dataclass(frozen=True, slots=True)
class RunInput:
    """What an AG-UI `RunAgentInput` asks for, ready for `agent.execute(...)`."""

    thread_id: str
    run_id: str
    prompt: Prompt | None  # the new user message, if the request has one
    history: list[Message]  # the conversation before it, as the client has it
    answers: dict[str, Reply]  # replies to tool calls the last run left waiting
    frontend_tools: list[dict[str, Any]]  # the client's tools, for this run
    state: Any = None
    context: list[dict[str, Any]] = field(default_factory=list)
    forwarded_props: Any = None
    body: dict[str, Any] = field(default_factory=dict)  # the whole request


def stream(agent: Agent, body: bytes | str | Mapping[str, Any], **execute_options: Any) -> ProtocolStream:
    """Runs the agent for an AG-UI request and returns its AG-UI event stream.

    `execute_options` go to `agent.execute` (`deps=...`). With `conversation_id=`, the
    history comes from the agent's history store instead of the request.
    """
    run_input = parse(body)
    history = None if "conversation_id" in execute_options else run_input.history or None
    run = agent.execute(
        run_input.prompt,
        history=history,
        answers=run_input.answers,
        frontend_tools=run_input.frontend_tools,
        **execute_options,
    )
    return run.ag_ui(run_input.thread_id, run_input.run_id)


def parse(body: bytes | str | Mapping[str, Any]) -> RunInput:
    """Reads an AG-UI `RunAgentInput`: `{threadId, runId, messages, tools?, resume?, state?, ...}`."""
    request = dict(body) if isinstance(body, Mapping) else json.loads(body)
    agui_messages = list(request.get("messages") or [])
    frontend_tools = list(request.get("tools") or [])
    frontend_tool_names = {tool["name"] for tool in frontend_tools}

    prompt = None
    if agui_messages and agui_messages[-1]["role"] == "user":
        prompt = _user_content(agui_messages.pop()["content"])

    # Replies come in `resume` (approvals, questions) or, for frontend tools, as
    # tool messages after the last assistant message. A new prompt replaces them.
    answers: dict[str, Reply] = {}
    kinds: dict[str, RequestKind] = {}  # what each answered call was waiting for
    resume = [] if prompt is not None else request.get("resume") or []
    for entry in resume:
        answers[entry["interruptId"]], kinds[entry["interruptId"]] = _resume_reply(entry)

    history: list[Message] = []
    names: dict[str, str] = {}  # tool call id -> tool name
    last_assistant = max((i for i, m in enumerate(agui_messages) if m["role"] == "assistant"), default=-1)
    for position, message in enumerate(agui_messages):  # system and developer messages are left out
        if message["role"] == "user":
            history.append({"role": "user", "content": _user_content(message["content"])})
        elif message["role"] == "assistant":
            tool_calls: list[ToolCall] = [
                {"id": call["id"], "name": call["function"]["name"], "arguments": _arguments(call)}
                for call in message.get("toolCalls") or []
            ]
            names.update((call["id"], call["name"]) for call in tool_calls)
            history.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls})
        elif message["role"] == "tool":
            call_id = message["toolCallId"]
            content = _text(message["content"])
            name = names.get(call_id, "")
            if prompt is None and position > last_assistant and name in frontend_tool_names:
                answers[call_id] = Deny(message["error"]) if message.get("error") else Answer(content)
                kinds[call_id] = "frontend"
                continue  # an answer: its waiting message is added below
            is_error = bool(message.get("error"))
            history.append(
                {"role": "tool", "tool_call_id": call_id, "name": name, "content": content, "is_error": is_error}
            )

    # Calls of the last assistant message without a result were waiting: give
    # each a waiting tool message, so the run continues them with `answers`.
    with_result = {message["tool_call_id"] for message in history if message["role"] == "tool"}
    for call in _last_tool_calls(history):
        if call["id"] not in with_result:
            kind = kinds.get(call["id"], "frontend" if call["name"] in frontend_tool_names else "input")
            history.append(_waiting_message(call, kind))

    return RunInput(
        thread_id=request["threadId"],
        run_id=request["runId"],
        prompt=prompt,
        history=history,
        answers=answers,
        frontend_tools=frontend_tools,
        state=request.get("state"),
        context=list(request.get("context") or []),
        forwarded_props=request.get("forwardedProps"),
        body=request,
    )


def _waiting_message(call: ToolCall, kind: RequestKind) -> ToolMessage:
    """A tool message for a call that waits for a person (see `human_input`)."""
    request: Request = {"kind": kind, "data": None, "metadata": None, "answer": None}
    return {"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": "", "requests": [request]}


def _last_tool_calls(history: list[Message]) -> list[ToolCall]:
    for message in reversed(history):
        if message["role"] == "assistant":
            return list(message.get("tool_calls") or [])
    return []


def _resume_reply(entry: dict[str, Any]) -> tuple[Reply, RequestKind]:
    """The reply in a `resume` entry, and the kind of request it answers."""
    payload = entry.get("payload")
    is_approval = isinstance(payload, Mapping) and isinstance(payload.get("approved"), bool)
    kind: RequestKind = "approval" if is_approval else "input"
    if entry.get("status") == "cancelled":
        return Deny("The user cancelled"), kind
    if is_approval and payload["approved"]:
        return Approve(arguments=payload.get("editedArgs")), kind
    if is_approval:
        return Deny(payload.get("reason") or "The user denied this tool call"), kind
    return Answer(payload), kind


def _arguments(call: dict[str, Any]) -> Any:
    arguments = call["function"].get("arguments") or "{}"
    return json.loads(arguments) if isinstance(arguments, str) else arguments


def _text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if part["type"] == "text")


def _user_content(content: str | list[dict[str, Any]]) -> str | list[ContentPart]:
    if isinstance(content, str):
        return content
    parts: list[ContentPart] = []
    for part in content:
        if part["type"] == "text":
            parts.append({"type": "text", "text": part["text"]})
        elif part["type"] in ("image", "document"):
            url = _source_url(part["source"])
            if part["type"] == "image":
                parts.append({"type": "image", "url": url})
            else:
                parts.append(
                    {"type": "file", "url": url, "filename": (part.get("metadata") or {}).get("filename", "file")}
                )
    return parts


def _source_url(source: dict[str, Any]) -> str:
    if source["type"] == "data":
        return f"data:{source['mimeType']};base64,{source['value']}"
    url = source["value"]
    # Never let a client make the server read a local path or an internal address.
    if source["type"] != "url" or not url.startswith("https://"):
        raise ValueError(f"Images and documents must be data or https:// URLs, got {url[:40]!r}")
    return url
