"""A tiny local server that imitates the streaming Chat Completions API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def text_response(text: str) -> list[dict[str, Any]]:
    """Chunks of a streamed answer containing only text (split in two pieces)."""
    middle = len(text) // 2
    return [
        {"choices": [{"delta": {"role": "assistant", "content": text[:middle]}}]},
        {"choices": [{"delta": {"content": text[middle:]}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    ]


def tool_calls_response(*calls: tuple[str, str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunks requesting tool calls `(call_id, name, arguments)`, with arguments split in pieces."""
    chunks: list[dict[str, Any]] = []
    for index, (call_id, name, arguments) in enumerate(calls):
        arguments_text = json.dumps(arguments)
        middle = len(arguments_text) // 2
        first = {"index": index, "id": call_id, "function": {"name": name, "arguments": arguments_text[:middle]}}
        rest = {"index": index, "function": {"arguments": arguments_text[middle:]}}
        chunks.append({"choices": [{"delta": {"tool_calls": [first]}}]})
        chunks.append({"choices": [{"delta": {"tool_calls": [rest]}}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    return chunks


class FakeOpenAI:
    """Answers each request with the next scripted response and records the requests."""

    def __init__(self) -> None:
        self.responses: list[list[dict[str, Any]] | tuple[int, str]] = []
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def close(self) -> None:
        self._server.shutdown()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                fake.requests.append(json.loads(body))
                fake.headers.append({name.lower(): value for name, value in self.headers.items()})
                response = fake.responses.pop(0)

                if isinstance(response, tuple):  # (status code, error body)
                    status, error_body = response
                    self.send_response(status)
                    self.end_headers()
                    self.wfile.write(error_body.encode())
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in response:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

        return Handler
