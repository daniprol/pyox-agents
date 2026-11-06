"""An agent with skills: instructions and tools the model loads only when a task needs them.

OPENAI_API_KEY=... uv run examples/skills.py

`examples/skills/` holds one skill in the Agent Skills format (a folder with a SKILL.md
and the files it points to). The other skill is made in code, with a tool that the model
only sees once it loads the skill.
"""

from pathlib import Path

from pyox_agents import Agent
from pyox_agents.skills import Skill, skills_from_dir


def get_forecast(place: str) -> str:
    """Get tomorrow's weather forecast for a place."""
    return f"{place}: light rain in the morning, then sunny, 14 °C"


weather = Skill(
    name="weather",
    description="Weather forecasts. Use when the weather matters for the answer.",
    instructions="Call get_forecast for the place, then say how the weather changes the plan.",
    tools=[get_forecast],  # hidden until the model loads the skill
)

agent = Agent(
    model="gpt-5-mini",
    instructions="You are a friendly outdoors guide. Be brief.",
    skills=[*skills_from_dir(Path(__file__).parent / "skills"), weather],
)

result = agent.run_sync("Plan me a hike in the Picos de Europa tomorrow. We are fit.")
for message in result.messages:
    for call in message.get("tool_calls", []):
        print(f"[{call['name']} {call['arguments']}]")
print(result.text)
