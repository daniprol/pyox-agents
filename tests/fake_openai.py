"""A real local HTTP server that imitates the streaming OpenAI APIs (Chat Completions and Responses).

Tests talk to it over HTTP exactly like they would talk to OpenAI, so the
whole Rust stack (HTTP, SSE decoding, parsing) is exercised without mocks.
"""

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Chunk = dict[str, Any]


@dataclass(frozen=True)
class StreamResponse:
    """A successful streamed answer: chunks sent as SSE events, then `[DONE]`.

    A chunk with a `"type"` (a Responses API event) is also named in an `event:`
    line, as OpenAI does. The Responses API does not end with `[DONE]`.
    """

    chunks: list[Chunk]
    delay_seconds: float = 0  # wait before answering, to test timeouts
    end_with_done: bool = True


@dataclass(frozen=True)
class ErrorResponse:
    """An HTTP error answer."""

    status: int
    body: str


@dataclass(frozen=True)
class DroppedConnection:
    """Close the connection without answering, like a server dropping an idle keep-alive connection."""


Response = StreamResponse | ErrorResponse | DroppedConnection


@dataclass(frozen=True)
class RecordedRequest:
    path: str  # e.g. "/v1/responses"
    body: dict[str, Any]
    headers: dict[str, str]  # names in lowercase


@dataclass
class FakeOpenAI:
    """Answers each request with the next queued response and records every request."""

    responses: list[Response] = field(default_factory=list)
    requests: list[RecordedRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def queue(self, *responses: Response) -> None:
        self.responses.extend(responses)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _next_response(self, request: RecordedRequest) -> Response:
        with self._lock:
            self.requests.append(request)
            if not self.responses:
                return ErrorResponse(500, "FakeOpenAI: no response queued for this request")
            return self.responses.pop(0)


def _handler_for(fake: FakeOpenAI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (silences request logs)
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            headers = {name.lower(): value for name, value in self.headers.items()}
            response = fake._next_response(RecordedRequest(path=self.path, body=body, headers=headers))

            if isinstance(response, DroppedConnection):
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return

            if isinstance(response, ErrorResponse):
                self.send_response(response.status)
                self.end_headers()
                self.wfile.write(response.body.encode())
                return

            time.sleep(response.delay_seconds)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in response.chunks:
                event_line = f"event: {chunk['type']}\n" if "type" in chunk else ""
                self.wfile.write(f"{event_line}data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            if response.end_with_done:
                self.wfile.write(b"data: [DONE]\n\n")

    return Handler
