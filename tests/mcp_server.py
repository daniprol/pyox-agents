"""A real MCP server for the tests, built with the official `mcp` package: a small room-booking service.

Tests connect to it in the same process (`McpTools(server)`), or run this file as a subprocess:
`python tests/mcp_server.py stdio` or `python tests/mcp_server.py http PORT`. Each tool exists for
one thing the tests check (structured results, approvals, questions to the user, images, errors...).
"""

import sys
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.resolve import Elicit, Resolve
from mcp.server.mcpserver.utilities.types import Image
from mcp_types import ToolAnnotations
from pydantic import BaseModel

# A 1x1 PNG.
PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da63f8ff"
    "ff3f0005fe02fea7d6a8b80000000049454e44ae426082"
)

server = MCPServer("rooms", instructions="Book meeting rooms.")

bookings: list[str] = []
"""Every booking made, so in-process tests can check what the server really did."""


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@server.tool(annotations=ToolAnnotations(destructive_hint=True))
def cancel_booking(booking_id: str) -> str:
    """Cancel a booking."""
    return f"Cancelled {booking_id}"


class Floor(BaseModel):
    floor: int


def ask_floor() -> Elicit[Floor]:
    return Elicit("Which floor?", Floor)


@server.tool()
def book_room(day: str, floor: Annotated[Floor, Resolve(ask_floor)]) -> str:
    """Book a meeting room. Asks the user which floor."""
    booking = f"{day} on floor {floor.floor}"
    bookings.append(booking)
    return f"Booked {booking}"


@server.tool(name="floor.plan", annotations=ToolAnnotations(read_only_hint=True))
def floor_plan() -> Image:
    """The floor plan, as an image."""
    return Image(data=PIXEL_PNG, format="png")


@server.tool()
def broken() -> str:
    """Always fails."""
    raise ToolError("The room service is down")


@server.tool()
def whoami(ctx: Context) -> str:
    """The Authorization header of the request (HTTP only)."""
    return (ctx.headers or {}).get("authorization", "nobody")


@server.tool()
def trace_parent(ctx: Context) -> str:
    """The W3C `traceparent` the client sent in the request's `_meta`."""
    meta = ctx.request_context.meta or {}
    return meta.get("traceparent", "none")


@server.prompt()
def review(day: str) -> str:
    """Review the bookings of a day."""
    return f"Review the bookings for {day}."


@server.resource("rooms://rules", mime_type="text/plain")
def rules() -> str:
    """The house rules."""
    return "No food in the rooms."


if __name__ == "__main__":
    if sys.argv[1] == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", port=int(sys.argv[2]))
