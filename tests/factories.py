"""Reusable builders for fake API responses and helpers to read run events."""

import json
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from fake_openai import ErrorResponse, JsonResponse, StreamResponse
from pyox_agents import AgentEvent, Run

DEFAULT_USAGE = {"prompt_tokens": 10, "completion_tokens": 5}

EventType = TypeVar("EventType")


def text_response(text: str, *, usage: dict[str, Any] = DEFAULT_USAGE) -> StreamResponse:
    """An answer with only text, streamed in two pieces to exercise reassembly."""
    middle = len(text) // 2
    return StreamResponse(
        [
            {"choices": [{"delta": {"role": "assistant", "content": text[:middle]}}]},
            {"choices": [{"delta": {"content": text[middle:]}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        ]
    )


def text_pieces_response(*pieces: str) -> StreamResponse:
    """An answer streamed exactly in the given pieces."""
    chunks: list[dict[str, Any]] = [{"choices": [{"delta": {"content": piece}}]} for piece in pieces]
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    return StreamResponse(chunks)


def tool_call(name: str, /, **arguments: Any) -> tuple[str, dict[str, Any]]:
    """One requested tool call, for `tool_calls_response`."""
    return name, arguments


def tool_calls_response(
    *calls: tuple[str, dict[str, Any]], text_pieces: tuple[str, ...] = (), usage: dict[str, Any] = DEFAULT_USAGE
) -> StreamResponse:
    """An answer requesting tool calls, optionally after some text. Call ids are `call_0`, `call_1`, ...

    The arguments JSON of each call is split in two chunks, as real APIs do.
    """
    chunks: list[dict[str, Any]] = [{"choices": [{"delta": {"content": piece}}]} for piece in text_pieces]
    for index, (name, arguments) in enumerate(calls):
        arguments_text = json.dumps(arguments)
        middle = len(arguments_text) // 2
        first_piece = {
            "index": index,
            "id": f"call_{index}",
            "function": {"name": name, "arguments": arguments_text[:middle]},
        }
        second_piece = {"index": index, "function": {"arguments": arguments_text[middle:]}}
        chunks.append({"choices": [{"delta": {"tool_calls": [first_piece]}}]})
        chunks.append({"choices": [{"delta": {"tool_calls": [second_piece]}}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    chunks.append({"choices": [], "usage": usage})
    return StreamResponse(chunks)


def error_response(status: int, message: str, *, code: str | None = None) -> ErrorResponse:
    """An API error. `code` is the machine-readable `error.code`, e.g. "previous_response_not_found"."""
    error: dict[str, Any] = {"message": message}
    if code is not None:
        error["code"] = code
    return ErrorResponse(status, json.dumps({"error": error}))


def decision_response(*answers: dict[str, Any], request_id: str = "req_1") -> JsonResponse:
    """A Decisions API answer, shaped like OpenAI's: `decision_response(predicate_answer("urgent", 0.9))`."""
    usage = {
        "input_tokens": 42,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 42,
    }
    body = {"model": "gpt-6-luna", "answers": list(answers), "usage": usage}
    return JsonResponse(body, headers={"x-request-id": request_id})


def predicate_answer(name: str, probability: float) -> dict[str, Any]:
    return {"type": "predicate", "name": name, "probability": probability}


async def collect_events(run: Run) -> list[AgentEvent]:
    return [event async for event in run]


def events_of_type(events: list[AgentEvent], event_type: type[EventType]) -> list[EventType]:
    return [event for event in events if isinstance(event, event_type)]


# Responses API ------------------------------------------------------------------------------------

DEFAULT_RESPONSES_USAGE = {"input_tokens": 10, "output_tokens": 5}


def responses_answer(
    text: str = "",
    *,
    tool_calls: tuple[tuple[str, dict[str, Any]], ...] = (),
    reasoning: tuple[str, ...] = (),
    response_id: str = "resp_1",
    usage: dict[str, Any] = DEFAULT_RESPONSES_USAGE,
) -> StreamResponse:
    """A streamed Responses API answer, shaped like OpenAI's.

    The text is streamed in two `response.output_text.delta` events, then `response.completed`
    carries the whole output: one reasoning item per `reasoning` summary (ids `rs_0`, `rs_1`, ...,
    encrypted content `encrypted-0`, ...), the message, then the tool calls (from `tool_call(...)`,
    call ids `call_0`, `call_1`, ...).
    """
    output: list[dict[str, Any]] = [
        {
            "type": "reasoning",
            "id": f"rs_{index}",
            "summary": [{"type": "summary_text", "text": summary}],
            "encrypted_content": f"encrypted-{index}",
        }
        for index, summary in enumerate(reasoning)
    ]
    if text:
        output.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
    for index, (name, arguments) in enumerate(tool_calls):
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{index}",
                "call_id": f"call_{index}",
                "name": name,
                "arguments": json.dumps(arguments),
            }
        )

    middle = len(text) // 2
    text_events = [{"type": "response.output_text.delta", "delta": piece} for piece in (text[:middle], text[middle:])]
    return StreamResponse(
        [
            {"type": "response.created", "response": {"id": response_id, "output": []}},
            *(text_events if text else []),
            {"type": "response.completed", "response": {"id": response_id, "output": output, "usage": usage}},
        ],
        end_with_done=False,
    )


# Web streams (`run.ai_sdk()`, `run.ag_ui(...)`) ---------------------------------------------------


async def read_text(stream: AsyncIterator[str]) -> str:
    return "".join([text async for text in stream])


def sse_data(text: str) -> list[Any]:
    """The `data:` payloads of a server-sent events text, parsed (`[DONE]` kept as a string)."""
    events = [event.removeprefix("data: ") for event in text.split("\n\n") if event]
    return [event if event == "[DONE]" else json.loads(event) for event in events]


def user_ui_message(text: str) -> dict[str, Any]:
    """A user message as AI SDK's `useChat` sends it."""
    return {"id": "u1", "role": "user", "parts": [{"type": "text", "text": text}]}
