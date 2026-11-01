"""A Django app (ASGI, async views) that streams an agent's answer, for the HTTP load test (`load_test.py`).

    AGENT=pydantic-ai uvicorn django_server:app --port 8000

The same endpoints as the FastAPI `server.py`, and the agents use async tools:
`POST /chat` with `{"prompt": "..."}` streams the answer text as it arrives.
`GET /stats` returns this process' peak memory and CPU time, read by the load test.
"""

import json
import os
import time
from collections.abc import AsyncIterator

import django
from django.conf import settings
from django.core.asgi import get_asgi_application
from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.urls import path

from libraries import DJANGO_LIBRARIES
from measure import peak_memory_mb  # also hides pydantic-ai's banner

AGENT = os.environ.get("AGENT", "pyox-agents")
MODEL_URL = os.environ.get("MODEL_URL", "http://127.0.0.1:4010/v1")  # the fake OpenAI server

stream_answer = DJANGO_LIBRARIES[AGENT](MODEL_URL)  # one agent for the whole app, shared by all requests


async def chat(request: HttpRequest) -> StreamingHttpResponse:
    pieces = stream_answer(json.loads(request.body)["prompt"])
    # Start the response only when the first piece of text is ready, so "time to first byte" is when the text starts.
    first_piece = await anext(pieces)

    async def all_pieces() -> AsyncIterator[str]:
        yield first_piece
        async for piece in pieces:
            yield piece

    return StreamingHttpResponse(all_pieces(), content_type="text/plain")


async def stats(request: HttpRequest) -> JsonResponse:
    return JsonResponse({"memory_peak_mb": peak_memory_mb(), "cpu_seconds": time.process_time()})


# A whole Django project in one file: no middleware (no sessions, no CSRF), no database.
settings.configure(
    DEBUG=False,
    SECRET_KEY="only-for-the-benchmark",
    ALLOWED_HOSTS=["127.0.0.1", "localhost"],
    ROOT_URLCONF=__name__,
    MIDDLEWARE=[],
)
urlpatterns = [path("chat", chat), path("stats", stats)]
django.setup()
app = get_asgi_application()
