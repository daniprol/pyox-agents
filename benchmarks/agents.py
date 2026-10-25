"""The agents we measure, all doing the same job.

Each `make_*` function returns `run(prompt) -> RunResult`. The run streams the
answer and goes through every event, like an app showing text as it arrives.
Libraries are imported inside the functions, so a process only loads the one it measures.
"""

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyox_agents.agent import TextChunks


@dataclass
class RunResult:
    text: str  # the final answer
    first_text_seconds: float  # time until the first piece of text reached our code


RunAgent = Callable[[str], Awaitable[RunResult]]

MODEL = "gpt-5-mini"  # any name works: the fake server ignores it
PROMPT = "You are a helpful assistant."


def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"Sunny, 21 degrees in {city}"


def make_pyox_agent(base_url: str, text_chunks: "TextChunks" = "delta") -> RunAgent:
    from pyox_agents import Agent, Client, RunFinished, TextDelta

    agent = Agent(
        client=Client(api_key="not-needed", base_url=base_url),
        model=MODEL,
        prompt=PROMPT,
        tools=[get_weather],
        text_chunks=text_chunks,
    )

    async def run(prompt: str) -> RunResult:
        start = time.perf_counter()
        first_text_seconds = 0.0
        final_text = ""
        async for event in agent.execute(prompt):
            if isinstance(event, TextDelta) and not first_text_seconds:
                first_text_seconds = time.perf_counter() - start
            elif isinstance(event, RunFinished):
                final_text = event.text
        return RunResult(final_text, first_text_seconds)

    return run


def make_pydantic_ai_agent(base_url: str) -> RunAgent:
    from pydantic_ai import Agent, AgentRunResultEvent, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    agent = Agent(
        OpenAIChatModel(MODEL, provider=OpenAIProvider(api_key="not-needed", base_url=base_url)),
        instructions=PROMPT,
        tools=[get_weather],
    )

    async def run(prompt: str) -> RunResult:
        start = time.perf_counter()
        first_text_seconds = 0.0
        final_text = ""
        async with agent.run_stream_events(prompt) as events:
            async for event in events:
                is_text = (isinstance(event, PartStartEvent) and isinstance(event.part, TextPart)) or (
                    isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta)
                )
                if is_text and not first_text_seconds:
                    first_text_seconds = time.perf_counter() - start
                elif isinstance(event, AgentRunResultEvent):
                    final_text = event.result.output
        return RunResult(final_text, first_text_seconds)

    return run


# Comparison 1: the same agent in each library.
LIBRARIES: dict[str, Callable[[str], RunAgent]] = {
    "pyox-agents": make_pyox_agent,
    "pydantic-ai": make_pydantic_ai_agent,
}

# Comparison 2: pyox-agents with each way of grouping streamed text (`text_chunks`).
TEXT_CHUNKS_MODES: list["TextChunks"] = ["delta", 10, "paragraph", "message"]
TEXT_CHUNKS: dict[str, Callable[[str], RunAgent]] = {
    f"text_chunks={mode!r}": partial(make_pyox_agent, text_chunks=mode) for mode in TEXT_CHUNKS_MODES
}

AGENTS = LIBRARIES | TEXT_CHUNKS
