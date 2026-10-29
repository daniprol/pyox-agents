"""A FastAPI app that streams an agent's answer, for the HTTP load test (`load_test.py`).

    AGENT=pydantic-ai uvicorn server:app --port 8000

`POST /chat` with `{"prompt": "..."}` streams the answer text as it arrives.
`GET /stats` returns this process' peak memory and CPU time, read by the load test.
"""

import os
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.types import Send

from libraries import LIBRARIES
from measure import peak_memory_mb  # also hides pydantic-ai's banner

AGENT = os.environ.get("AGENT", "pyox-agents")
MODEL_URL = os.environ.get("MODEL_URL", "http://127.0.0.1:4010/v1")  # the fake OpenAI server

stream_answer = LIBRARIES[AGENT](MODEL_URL)  # one agent for the whole app, shared by all requests
app = FastAPI()


class ChatRequest(BaseModel):
    prompt: str


class TextResponse(StreamingResponse):
    """Streams text, but starts the response only when the first piece of text is ready.

    So "time to first byte" is measured when the text starts. The whole agent run stays in the
    response's task: Microsoft Agent Framework fails if a run moves to another asyncio task midway.
    """

    async def stream_response(self, send: Send) -> None:
        pieces = self.body_iterator
        first_piece = await anext(pieces)

        async def all_pieces() -> AsyncIterator[str | bytes | memoryview]:
            yield first_piece
            async for piece in pieces:
                yield piece

        self.body_iterator = all_pieces()
        await super().stream_response(send)


@app.post("/chat")
async def chat(request: ChatRequest) -> TextResponse:
    return TextResponse(stream_answer(request.prompt), media_type="text/plain")


@app.get("/stats")
def stats() -> dict[str, float]:
    return {"memory_peak_mb": peak_memory_mb(), "cpu_seconds": time.process_time()}
