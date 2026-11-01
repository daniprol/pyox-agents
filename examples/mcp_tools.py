"""An agent using the tools of an MCP server. The server is in this same file, for a self-contained example.

OPENAI_API_KEY=... uv run examples/mcp_tools.py

The script starts itself again with `serve` as a local MCP server (over stdin/stdout),
then gives its tools to an agent. Needs `pip install "pyox-agents[mcp]"`.
"""

import asyncio
import sys
from typing import Annotated

from pydantic import BaseModel

from pyox_agents import Agent, Answer, Approve, Deny, InputRequested, Reply
from pyox_agents.mcp import McpTools

# The server ------------------------------------------------------------------------------------


def serve() -> None:
    from mcp.server import MCPServer
    from mcp.server.mcpserver.resolve import Elicit, Resolve
    from mcp_types import ToolAnnotations

    server = MCPServer("picnics")

    class People(BaseModel):
        people: int

    def ask_people() -> Elicit[People]:
        return Elicit("How many people are coming?", People)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True))
    def forecast(city: str) -> str:
        """The weather forecast for tomorrow in a city."""
        return f"Sunny and 24°C in {city}"

    @server.tool()
    def book_picnic(city: str, people: Annotated[People, Resolve(ask_people)]) -> str:
        """Book a picnic in a city's park. Asks the user how many people are coming."""
        return f"Picnic for {people.people} booked in {city}"

    server.run("stdio")


# The agent -------------------------------------------------------------------------------------


def ask_in_the_terminal(request: InputRequested) -> Reply:
    if request.kind == "approval":
        allowed = input(f"Allow {request.tool_name} {request.arguments}? [y/n] ") == "y"
        return Approve() if allowed else Deny()
    # A question from the MCP server: `data` has its message and the fields it wants.
    return Answer({"people": int(input(f"{request.data['message']} "))})


async def main() -> None:
    # Tools the server did not mark as read-only (here `book_picnic`) wait for approval.
    picnics = McpTools.stdio(sys.executable, [__file__, "serve"], requires_approval="unless_read_only")
    async with picnics:
        agent = Agent(model="gpt-5-mini", instructions="Be brief.", tools=picnics.tools)
        result = await agent.run(
            "Is tomorrow good for a picnic in Vigo? If so, book one.", on_request=ask_in_the_terminal
        )
        print(result.text)


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        serve()
    else:
        asyncio.run(main())
