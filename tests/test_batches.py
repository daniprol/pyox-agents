"""`run.batches()`: every event already waiting, in one step."""

import asyncio

import pytest

from conftest import MakeAgent
from factories import error_response, text_pieces_response, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import AgentEvent, ApiError, EventBatches, RunFinished, TextDelta, ToolFinished, ToolStarted


def add(a: int, b: int) -> int:
    return a + b


async def test_batches_hold_every_event_in_order(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("add", a=1, b=2)), text_pieces_response("The ", "answer ", "is 3"))

    batches = [events async for events in make_agent(tools=[add]).execute("1 + 2?").batches()]

    events = [event for batch in batches for event in batch]
    assert all(batches)  # never an empty batch
    assert [type(event) for event in events] == [
        ToolStarted,
        ToolFinished,
        TextDelta,
        TextDelta,
        TextDelta,
        RunFinished,
    ]
    assert isinstance(events[-1], RunFinished)
    assert events[-1].text == "The answer is 3"


async def test_events_that_piled_up_come_in_one_batch(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_pieces_response(*(f"{number} " for number in range(20))))
    batches = make_agent().execute("Count").batches()

    first = await anext(batches)  # starts the run
    await asyncio.sleep(0.2)  # meanwhile the whole answer arrives
    rest = [events async for events in batches]

    assert len(first) + sum(len(events) for events in rest) == 21
    assert len(rest) == 1  # everything that waited came at once


async def test_a_failing_run_raises_after_its_events(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("add", a=1, b=2)), error_response(500, "Down"))
    batches = make_agent(tools=[add]).execute("1 + 2?").batches()

    events: list[AgentEvent] = []
    with pytest.raises(ApiError, match="Down"):
        await read_into(events, batches)

    assert [type(event) for event in events] == [ToolStarted, ToolFinished]


async def test_a_text_run_matches_plain_iteration(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hello there"), text_response("Hello there"))
    agent = make_agent()

    one_by_one = [event async for event in agent.execute("Hi")]
    batched = [event async for batch in agent.execute("Hi").batches() for event in batch]

    assert [repr(event) for event in one_by_one] == [repr(event) for event in batched]


async def read_into(events: list[AgentEvent], batches: EventBatches) -> None:
    async for batch in batches:
        events.extend(batch)
