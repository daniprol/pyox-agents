"""The same agent written with each library.

Each `make_*` function returns `run(prompt) -> final text`. The run streams the
answer and goes through every event, like an app showing tokens as they arrive.
Libraries are imported inside the functions, so a process only loads the one it measures.
"""

from collections.abc import Awaitable, Callable

RunAgent = Callable[[str], Awaitable[str]]

MODEL = "gpt-5-mini"  # any name works: the fake server ignores it
PROMPT = "You are a helpful assistant."


def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"Sunny, 21 degrees in {city}"


def make_pyox_agent(base_url: str) -> RunAgent:
    from pyox_agents import Agent, Client, RunFinished

    agent = Agent(
        client=Client(api_key="not-needed", base_url=base_url),
        model=MODEL,
        prompt=PROMPT,
        tools=[get_weather],
    )

    async def run(prompt: str) -> str:
        final_text = ""
        async for event in agent.execute(prompt):
            if isinstance(event, RunFinished):
                final_text = event.text
        return final_text

    return run


def make_pydantic_ai_agent(base_url: str) -> RunAgent:
    from pydantic_ai import Agent, AgentRunResultEvent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    agent = Agent(
        OpenAIChatModel(MODEL, provider=OpenAIProvider(api_key="not-needed", base_url=base_url)),
        instructions=PROMPT,
        tools=[get_weather],
    )

    async def run(prompt: str) -> str:
        final_text = ""
        async with agent.run_stream_events(prompt) as events:
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    final_text = event.result.output
        return final_text

    return run


LIBRARIES: dict[str, Callable[[str], RunAgent]] = {
    "pyox-agents": make_pyox_agent,
    "pydantic-ai": make_pydantic_ai_agent,
}
