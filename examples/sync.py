"""An agent in plain sync code: no `async`, no event loop.

OPENAI_API_KEY=... uv run examples/sync.py
"""

from typing import Annotated

from pyox_agents import Agent, TextDelta, tool


@tool
def get_weather(city: Annotated[str, "City name"]) -> str:
    """Get the current weather in a city."""
    return f"Sunny, 21 °C in {city}"  # runs on this thread, like any function you call


agent = Agent(model="gpt-5-mini", instructions="You are a concise assistant.", tools=[get_weather])

# The whole answer at once.
result = agent.run_sync("What's the weather in Vigo?")
print(result.text)

# Or streamed, event by event.
for event in agent.execute_sync("And in Lugo?", history=result.messages):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
print()
