"""A planner with a researcher subagent, sharing a todo list (the run's state).

OPENAI_API_KEY=... uv run examples/subagents.py
"""

import asyncio

from pyox_agents import Agent, RunContext, StateChanged, SubagentFinished, SubagentStarted, TextDelta, tool


@tool
def find_places(city: str) -> list[str]:
    """Find places worth a visit in a city."""
    return [f"The old town of {city}", f"The market of {city}"]  # a real tool would search the web


@tool
def add_todo(ctx: RunContext, text: str) -> str:
    """Add an item to the trip's todo list."""
    ctx.state["todos"].append(text)
    return f"Added {text!r}"


researcher = Agent(
    model="gpt-5-mini",
    prompt="You find places to visit. Answer with a short list.",
    tools=[find_places],
)
planner = Agent(
    model="gpt-5-mini",
    prompt="You plan trips. Ask `research` for places, then add each one to the todo list.",
    tools=[add_todo, researcher.as_tool("research", "Find places to visit in a city.", state="read")],
)


async def main() -> None:
    async for event in planner.execute("Plan a day in Vigo", state={"todos": []}):
        match event:
            case SubagentStarted(name=name, task=task):
                print(f"[{name}] {task}")
            case SubagentFinished(name=name):
                print(f"[{name}] done")
            case StateChanged(state=state):
                print(f"todos: {state['todos']}")
            case TextDelta(text=text, subagent_id=None):  # only the planner's own text
                print(text, end="", flush=True)
    print()


asyncio.run(main())
