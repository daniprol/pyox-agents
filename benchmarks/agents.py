"""The agents we measure, all doing the same job.

Each `make_*` function returns `stream(prompt)`: an async iterator over the pieces
of answer text, as they arrive. Both benchmarks use it: `measure.py` calls it
directly, `server.py` sends the pieces to an HTTP client.
Libraries are imported inside the functions, so a process only loads the one it measures.
"""

from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyox_agents.agent import TextChunks

StreamText = Callable[[str], AsyncIterator[str]]

MODEL = "gpt-5-mini"  # any name works: the fake server ignores it
PROMPT = "You are a helpful assistant."


def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"Sunny, 21 degrees in {city}"


def make_pyox_agent(base_url: str, text_chunks: "TextChunks" = "delta") -> StreamText:
    from pyox_agents import Agent, Client, TextDelta

    agent = Agent(
        client=Client(api_key="not-needed", base_url=base_url),
        model=MODEL,
        prompt=PROMPT,
        tools=[get_weather],
        text_chunks=text_chunks,
    )

    async def stream(prompt: str) -> AsyncIterator[str]:
        async for event in agent.execute(prompt):
            if isinstance(event, TextDelta):
                yield event.text

    return stream


def make_pydantic_ai_agent(base_url: str) -> StreamText:
    from pydantic_ai import Agent, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    agent = Agent(
        OpenAIChatModel(MODEL, provider=OpenAIProvider(api_key="not-needed", base_url=base_url)),
        instructions=PROMPT,
        tools=[get_weather],
    )

    async def stream(prompt: str) -> AsyncIterator[str]:
        async with agent.run_stream_events(prompt) as events:
            async for event in events:
                # Text arrives as the start of a text part, then as deltas of that part.
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    yield event.part.content
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    yield event.delta.content_delta

    return stream


# Comparison 1: the same agent in each library.
LIBRARIES: dict[str, Callable[[str], StreamText]] = {
    "pyox-agents": make_pyox_agent,
    "pydantic-ai": make_pydantic_ai_agent,
}

# Comparison 2: pyox-agents with each way of grouping streamed text (`text_chunks`).
TEXT_CHUNKS_MODES: list["TextChunks"] = ["delta", 10, "paragraph", "message"]
TEXT_CHUNKS: dict[str, Callable[[str], StreamText]] = {
    f"text_chunks={mode!r}": partial(make_pyox_agent, text_chunks=mode) for mode in TEXT_CHUNKS_MODES
}

AGENTS = LIBRARIES | TEXT_CHUNKS
