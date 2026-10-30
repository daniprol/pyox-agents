"""A real local HTTP server that imitates the streaming OpenAI APIs (Chat Completions and Responses).

Tests talk to it over HTTP exactly like they would talk to OpenAI, so the
whole Rust stack (HTTP, SSE decoding, parsing) is exercised without mocks.

It also speaks the Responses API's WebSocket mode: a `GET /v1/responses` that
asks to upgrade becomes a WebSocket connection, and each `response.create`
message on it is answered with the next queued response, its events sent as
WebSocket messages instead of SSE events. Every request records the number of
the connection it came on, so tests can check which requests shared one.
"""

import base64
import hashlib
import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BufferedIOBase
from typing import Any, Literal

Chunk = dict[str, Any]


@dataclass(frozen=True)
class StreamResponse:
    """A successful streamed answer: chunks sent as SSE events, then `[DONE]`.

    A chunk with a `"type"` (a Responses API event) is also named in an `event:`
    line, as OpenAI does. The Responses API does not end with `[DONE]`.
    """

    chunks: list[Chunk]
    delay_seconds: float = 0  # wait before answering, to test timeouts
    end_with_done: bool = True  # never sent over WebSocket
    chunk_delay_seconds: float = 0  # wait between chunks, to stop a run in the middle
    close_after_chunks: bool = False  # over WebSocket, close the connection (an HTTP stream always ends)


@dataclass(frozen=True)
class ErrorResponse:
    """An HTTP error answer.

    Over WebSocket it becomes an error event: `{"type": "error", "status": status, **body}`.
    """

    status: int
    body: str


@dataclass(frozen=True)
class DroppedConnection:
    """Close the connection without answering, like a server dropping an idle keep-alive connection.

    Works the same over WebSocket.
    """


Response = StreamResponse | ErrorResponse | DroppedConnection


@dataclass(frozen=True)
class RecordedRequest:
    path: str  # e.g. "/v1/responses"
    body: dict[str, Any]  # over WebSocket, the `response.create` message
    headers: dict[str, str]  # names in lowercase; over WebSocket, those of the upgrade request
    transport: Literal["http", "websocket"] = "http"
    connection: int = 0  # 1 for the first connection the server accepted, 2 for the next...


@dataclass
class FakeOpenAI:
    """Answers each request with the next queued response and records every request.

    `websocket_refusal_status` (e.g. 404) answers every WebSocket upgrade with that
    status, like a server without WebSocket mode. `websocket_upgrades` records the
    headers of every upgrade request, accepted or not, and `closed_websockets` the
    WebSocket connections that have ended.
    """

    responses: list[Response] = field(default_factory=list)
    requests: list[RecordedRequest] = field(default_factory=list)
    websocket_refusal_status: int | None = None
    websocket_upgrades: list[dict[str, str]] = field(default_factory=list)
    closed_websockets: list[int] = field(default_factory=list)  # connection numbers

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._connections_accepted = 0
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

    def _new_connection_number(self) -> int:
        with self._lock:
            self._connections_accepted += 1
            return self._connections_accepted


def _handler_for(fake: FakeOpenAI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (silences request logs)
            pass

        def setup(self) -> None:
            # One handler serves one TCP connection (and its keep-alive requests).
            super().setup()
            self.connection_number = fake._new_connection_number()

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            headers = {name.lower(): value for name, value in self.headers.items()}
            request = RecordedRequest(path=self.path, body=body, headers=headers, connection=self.connection_number)
            response = fake._next_response(request)

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
                time.sleep(response.chunk_delay_seconds)
                event_line = f"event: {chunk['type']}\n" if "type" in chunk else ""
                self.wfile.write(f"{event_line}data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            if response.end_with_done:
                self.wfile.write(b"data: [DONE]\n\n")

        def do_GET(self) -> None:
            """Opens a WebSocket connection, then answers each `response.create` message on it."""
            headers = {name.lower(): value for name, value in self.headers.items()}
            if headers.get("upgrade", "").lower() != "websocket":
                self.send_error(404)
                return
            fake.websocket_upgrades.append(headers)
            if fake.websocket_refusal_status is not None:
                self.send_response(fake.websocket_refusal_status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", websocket_accept_key(headers["sec-websocket-key"]))
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True  # when this method returns, the connection is closed
            try:
                while (message := read_websocket_message(self.rfile, self.wfile)) is not None:
                    request = RecordedRequest(
                        path=self.path,
                        body=json.loads(message),
                        headers=headers,
                        transport="websocket",
                        connection=self.connection_number,
                    )
                    if not self._answer_over_websocket(fake._next_response(request)):
                        return
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client closed the connection (e.g. a cancelled run)
            finally:
                fake.closed_websockets.append(self.connection_number)

        def _answer_over_websocket(self, response: Response) -> bool:
            """Sends a response as WebSocket messages. False when the connection must close."""
            if isinstance(response, DroppedConnection):
                self.connection.shutdown(socket.SHUT_RDWR)
                return False
            if isinstance(response, ErrorResponse):
                try:
                    details = json.loads(response.body)
                except json.JSONDecodeError:
                    details = {"error": {"message": response.body}}
                send_websocket_text(self.wfile, json.dumps({"type": "error", "status": response.status, **details}))
                return True
            time.sleep(response.delay_seconds)
            for chunk in response.chunks:
                time.sleep(response.chunk_delay_seconds)
                send_websocket_text(self.wfile, json.dumps(chunk))
            return not response.close_after_chunks

    return Handler


# WebSocket framing (RFC 6455), just what the tests need -------------------------------------------

_TEXT, _CLOSE, _PING, _PONG = 0x1, 0x8, 0x9, 0xA


def websocket_accept_key(key: str) -> str:
    """The `Sec-WebSocket-Accept` answer to a client's `Sec-WebSocket-Key`."""
    digest = hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
    return base64.b64encode(digest).decode()


def read_websocket_message(reader: BufferedIOBase, writer: BufferedIOBase) -> str | None:
    """The next text message from the client, or None once it closes the connection. Answers pings."""
    payload = b""
    while True:
        header = reader.read(2)
        if len(header) < 2:
            return None
        is_final, opcode = header[0] & 0x80, header[0] & 0x0F
        length = header[1] & 0x7F
        if length == 126:
            (length,) = struct.unpack("!H", reader.read(2))
        elif length == 127:
            (length,) = struct.unpack("!Q", reader.read(8))
        mask = reader.read(4)  # client messages are always masked
        data = bytes(byte ^ mask[index % 4] for index, byte in enumerate(reader.read(length)))
        if opcode == _CLOSE:
            return None
        if opcode == _PING:
            _send_frame(writer, _PONG, data)
            continue
        if opcode == _PONG:
            continue
        payload += data  # a text message, or the next piece of one
        if is_final:
            return payload.decode()


def send_websocket_text(writer: BufferedIOBase, text: str) -> None:
    _send_frame(writer, _TEXT, text.encode())


def _send_frame(writer: BufferedIOBase, opcode: int, data: bytes) -> None:
    # Server messages are not masked. The length takes 1, 3 or 9 bytes.
    if len(data) < 126:
        header = struct.pack("!BB", 0x80 | opcode, len(data))
    elif len(data) < 1 << 16:
        header = struct.pack("!BBH", 0x80 | opcode, 126, len(data))
    else:
        header = struct.pack("!BBQ", 0x80 | opcode, 127, len(data))
    writer.write(header + data)
    writer.flush()
