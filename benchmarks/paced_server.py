"""A fake OpenAI Chat Completions server that streams like a real model: 100 pieces, 10 ms apart.

    python paced_server.py 4030

Used by `stream_bench.py`. Every request gets the same answer, whatever it asks.
"""

import asyncio
import json
import sys

PIECES = 100
PAUSE_S = 0.01


def sse(data: dict) -> bytes:
    """One server-sent event, as one chunk of a chunked HTTP response."""
    event = f"data: {json.dumps(data)}\n\n".encode()
    return f"{len(event):x}\r\n".encode() + event + b"\r\n"


def chunk(delta: dict, finish_reason: str | None = None) -> dict:
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "m", "choices": [choice]}


async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while True:  # one connection can carry many requests
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except asyncio.IncompleteReadError:
            return  # the client closed the connection
        length = next(
            int(line.split(b":")[1]) for line in head.lower().split(b"\r\n") if line.startswith(b"content-length")
        )
        await reader.readexactly(length)
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        for index in range(PIECES):
            writer.write(sse(chunk({"content": f"word{index} "})))
            await writer.drain()
            await asyncio.sleep(PAUSE_S)
        writer.write(sse(chunk({}, "stop")))
        done = b"data: [DONE]\n\n"
        writer.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")  # and the response's end
        await writer.drain()


async def main() -> None:
    server = await asyncio.start_server(answer, "127.0.0.1", int(sys.argv[1]))
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
