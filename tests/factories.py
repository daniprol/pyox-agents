"""Reusable builders for fake API responses and helpers to read run events."""

import json
from typing import Any, TypeVar

from fake_openai import ErrorResponse, StreamResponse
from pyox_agents import AgentEvent, Run

DEFAULT_USAGE = {"prompt_tokens": 10, "completion_tokens": 5}

EventType = TypeVar("EventType")


def text_response(text: str, *, usage: dict[str, int] = DEFAULT_USAGE) -> StreamResponse:
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


def tool_call(name: str, **arguments: Any) -> tuple[str, dict[str, Any]]:
    """One requested tool call, for `tool_calls_response`."""
    return name, arguments


def tool_calls_response(*calls: tuple[str, dict[str, Any]]) -> StreamResponse:
    """An answer requesting tool calls. Call ids are `call_0`, `call_1`, ...

    The arguments JSON of each call is split in two chunks, as real APIs do.
    """
    chunks: list[dict[str, Any]] = []
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
