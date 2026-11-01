"""Get a dataclass back from an agent, instead of text.

OPENAI_API_KEY=... uv run examples/structured_output.py
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated, Literal

from pyox_agents import Agent


@dataclass
class Ticket:
    team: Literal["billing", "support", "sales"]
    summary: Annotated[str, "One sentence"]
    urgent: bool
    order_id: str | None  # the model answers null when the message has none


agent = Agent(model="gpt-5-mini", instructions="Turn customer messages into tickets.", output_type=Ticket)


async def main() -> None:
    result = await agent.run("My order 1234 arrived broken and I need it for tomorrow's wedding!")
    ticket = result.output  # a `Ticket`, or None if the model refused or was cut off (see `result.stop_reason`)
    print(ticket)


asyncio.run(main())
