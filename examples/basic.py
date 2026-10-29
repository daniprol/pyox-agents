"""Stream an answer from an agent that has a Python tool and a built-in Rust tool.

OPENAI_API_KEY=... uv run examples/basic.py
"""

import asyncio
from typing import Annotated, Literal

from pyox_agents import Agent, BuiltinTool, RunFinished, TextDelta, ToolFinished, ToolStarted, tool


@tool
def get_weather(city: Annotated[str, "City name"], unit: Literal["C", "F"] = "C") -> str:
    """Get the current weather in a city."""
    return f"Sunny, 21 °{unit} in {city}"


agent = Agent(
    model="gpt-5-mini",
    prompt="You are a concise assistant.",
    tools=[get_weather, BuiltinTool("write_file")],
)


async def main() -> None:
    async for event in agent.execute("What's the weather in Vigo? Save it to weather.txt"):
        match event:
            case TextDelta(text=text):
                print(text, end="", flush=True)
            case ToolStarted(name=name, arguments=arguments):
                print(f"\n[calling {name} with {arguments}]")
            case ToolFinished(name=name, result=result):
                print(f"[{name} returned {result!r}]")
            case RunFinished(usage=usage):
                print(f"\n\nTokens: {usage.input_tokens} in, {usage.output_tokens} out")


asyncio.run(main())
