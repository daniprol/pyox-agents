"""A fake OpenAI Responses API, over HTTP and WebSocket, for the WebSocket benchmark (`websocket_bench.py`).

    MODEL_DELAY_MS=0 uvicorn responses_server:app --port 4020

- `POST /v1/responses` streams server-sent events; a WebSocket on `/v1/responses` answers each
  `response.create` message with the same events, one per WebSocket message.
- The prompt says how many tool calls the run makes, one per turn ("... 5 tool calls ..."). Each turn calls
  `get_weather` once; after the last tool result the model streams `TOOL_ANSWER` (about 4 characters per event).
- Over WebSocket, a request with `previous_response_id` continues a response made on the same connection, like
  OpenAI's per-connection cache (an unknown id gets `previous_response_not_found`). It cannot reproduce the real
  server's speedup: it only measures what each library does on its side.
- `MODEL_DELAY_MS` waits before each answer, like a model thinking.
- `GET /stats` returns the requests, WebSocket connections and request bytes seen so far.
"""

import asyncio
import itertools
import json
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from measure import TOOL_ANSWER

MODEL_DELAY_SECONDS = float(os.environ.get("MODEL_DELAY_MS", "0")) / 1000
CHARACTERS_PER_EVENT = 4

Event = dict[str, Any]

stats = {"requests": 0, "websocket_connections": 0, "request_bytes": 0}
response_ids = itertools.count(1)


def answer_events(body: dict[str, Any], wanted: int, tool_calls_done: int) -> tuple[list[Event], int]:
    """The events of the next answer, and how many tool calls the conversation has made after it."""
    response_id = f"resp_{next(response_ids)}"
    if tool_calls_done < wanted:
        call_id = f"call_{tool_calls_done}"
        item = {
            "type": "function_call",
            "id": f"fc_{response_id}",
            "call_id": call_id,
            "name": "get_weather",
            "arguments": json.dumps({"city": "Vigo"}),
            "status": "completed",
        }
        events = [
            {"type": "response.output_item.added", "output_index": 0, "item": {**item, "arguments": ""}},
            {"type": "response.function_call_arguments.delta", "item_id": item["id"], "output_index": 0}
            | {"delta": item["arguments"]},
            {"type": "response.function_call_arguments.done", "item_id": item["id"], "output_index": 0}
            | {"arguments": item["arguments"]},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
        ]
        tool_calls_done += 1
    else:
        message_id = f"msg_{response_id}"
        part = {"type": "output_text", "text": TOOL_ANSWER, "annotations": []}
        item = {"type": "message", "id": message_id, "role": "assistant", "status": "completed", "content": [part]}
        where = {"item_id": message_id, "output_index": 0, "content_index": 0}
        pieces = [TOOL_ANSWER[i : i + CHARACTERS_PER_EVENT] for i in range(0, len(TOOL_ANSWER), CHARACTERS_PER_EVENT)]
        events = [
            {"type": "response.output_item.added", "output_index": 0, "item": {**item, "content": []}},
            {"type": "response.content_part.added", **where, "part": {**part, "text": ""}},
            *({"type": "response.output_text.delta", **where, "delta": piece, "logprobs": []} for piece in pieces),
            {"type": "response.output_text.done", **where, "text": TOOL_ANSWER, "logprobs": []},
            {"type": "response.content_part.done", **where, "part": part},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
        ]
    completed = response_object(response_id, body, "completed", [item])
    events = [
        {"type": "response.created", "response": response_object(response_id, body, "in_progress", [])},
        *events,
        {"type": "response.completed", "response": completed},
    ]
    return [{**event, "sequence_number": number} for number, event in enumerate(events)], tool_calls_done


def wanted_tool_calls(body: dict[str, Any]) -> int:
    """How many tool calls the prompt asks for ("... 5 tool calls ..."), 0 if it does not say."""
    match = re.search(r"(\d+) tool calls?", json.dumps(body.get("input")))
    return int(match.group(1)) if match else 0


def response_object(response_id: str, body: dict[str, Any], status: str, output: list[Event]) -> Event:
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": body.get("model", "gpt-5-mini"),
        "status": status,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": {},
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 120,
        },
    }


def tool_results_in(body: dict[str, Any]) -> int:
    return sum(1 for item in body.get("input", []) if item.get("type") == "function_call_output")


async def http_responses(request: Request) -> StreamingResponse:
    raw_body = await request.body()
    stats["requests"] += 1
    stats["request_bytes"] += len(raw_body)
    body = json.loads(raw_body)
    # Over HTTP every library sends the whole conversation (no `store`): count its tool results.
    events, _ = answer_events(body, wanted_tool_calls(body), tool_results_in(body))

    async def stream() -> AsyncIterator[str]:
        await asyncio.sleep(MODEL_DELAY_SECONDS)
        for event in events:
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


async def websocket_responses(websocket: WebSocket) -> None:
    await websocket.accept()
    stats["websocket_connections"] += 1
    # This connection's cache: response id -> (tool calls wanted, tool calls made after it).
    cache: dict[str, tuple[int, int]] = {}
    try:
        while True:
            message = await websocket.receive_text()
            stats["requests"] += 1
            stats["request_bytes"] += len(message)
            body = json.loads(message)
            previous_response_id = body.get("previous_response_id")
            if previous_response_id is None:
                wanted, tool_calls_done = wanted_tool_calls(body), tool_results_in(body)
            elif previous_response_id in cache:
                wanted, tool_calls_done = cache[previous_response_id]
            else:
                error = {"code": "previous_response_not_found", "message": "Previous response not found."}
                await websocket.send_text(json.dumps({"type": "error", "status": 400, "error": error}))
                continue
            events, tool_calls_done = answer_events(body, wanted, tool_calls_done)
            cache[events[-1]["response"]["id"]] = (wanted, tool_calls_done)
            await asyncio.sleep(MODEL_DELAY_SECONDS)
            for event in events:
                await websocket.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass


async def get_stats(request: Request) -> JSONResponse:
    return JSONResponse(stats)


app = Starlette(
    routes=[
        Route("/v1/responses", http_responses, methods=["POST"]),
        WebSocketRoute("/v1/responses", websocket_responses),
        Route("/stats", get_stats),
    ]
)
