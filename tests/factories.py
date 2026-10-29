"""Reusable builders for fake API responses and helpers to read run events."""

import json
from typing import Any, TypeVar

from fake_openai import ErrorResponse, StreamResponse
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


def tool_calls_response(*calls: tuple[str, dict[str, Any]], text_pieces: tuple[str, ...] = ()) -> StreamResponse:
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
    return StreamResponse(chunks)


def error_response(status: int, message: str) -> ErrorResponse:
    return ErrorResponse(status, json.dumps({"error": {"message": message}}))


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
