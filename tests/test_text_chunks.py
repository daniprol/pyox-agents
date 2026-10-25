"""Grouping streamed text into bigger `TextDelta` events (the `text_chunks` option)."""

import pytest

from conftest import MakeAgent
from factories import collect_events, events_of_type, text_pieces_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import AgentEvent, TextChunks, TextDelta, ToolStarted

MARKDOWN_PIECES = ("Intro ", "line.\n", "\n", "```py\nx = 1\n", "\n", "y = 2\n```\n", "Last", " words.")


def texts(events: list[AgentEvent]) -> list[str]:
    return [event.text for event in events_of_type(events, TextDelta)]


@pytest.mark.parametrize(
    ("text_chunks", "expected"),
    [
        ("delta", list(MARKDOWN_PIECES)),
        (1, list(MARKDOWN_PIECES)),
        (3, ["Intro line.\n\n", "```py\nx = 1\n\ny = 2\n```\n", "Last words."]),
        ("paragraph", ["Intro line.\n\n", "```py\nx = 1\n\ny = 2\n```\n", "Last words."]),
        ("message", ["".join(MARKDOWN_PIECES)]),
    ],
)
async def test_modes(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, text_chunks: TextChunks, expected: list[str]
) -> None:
    fake_openai.queue(text_pieces_response(*MARKDOWN_PIECES))

    events = await collect_events(make_agent(text_chunks=text_chunks).execute("Hi"))

    assert texts(events) == expected


async def test_text_is_sent_before_tool_calls(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("ping"), text_pieces=("Let me ", "check.")),
        text_pieces_response("All ", "good."),
    )

    events = await collect_events(make_agent(tools=[lambda: "pong"], text_chunks="message").execute("Hi"))

    first_text, tool_started, second_text = events[0], events[1], events[3]
    assert isinstance(first_text, TextDelta)
    assert first_text.text == "Let me check."
    assert isinstance(tool_started, ToolStarted)
    assert isinstance(second_text, TextDelta)
    assert second_text.text == "All good."


async def test_execute_overrides_the_agent_setting(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_pieces_response("a", "b", "c"))

    events = await collect_events(make_agent(text_chunks="delta").execute("Hi", text_chunks="message"))

    assert texts(events) == ["abc"]


async def test_run_result_is_the_same_in_every_mode(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_pieces_response(*MARKDOWN_PIECES))

    result = await make_agent(text_chunks=2).run("Hi")

    assert result.text == "".join(MARKDOWN_PIECES)


@pytest.mark.parametrize("invalid", [0, "word", -1])
def test_invalid_values(make_agent: MakeAgent, invalid: object) -> None:
    with pytest.raises(ValueError, match="text_chunks"):
        make_agent(text_chunks=invalid)
