"""Tools from a separately compiled Rust package (examples/rust_tool_package, built by `uv sync`)."""

from pathlib import Path

import fast_tools

from conftest import MakeAgent
from factories import collect_events, events_of_type, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import ToolFinished


def test_schema_and_description_come_from_rust() -> None:
    tool = fast_tools.most_common_words

    assert tool.name == "most_common_words"
    assert tool.description == "Find the most frequent words in a text, with how many times each appears."
    assert tool.parameters["required"] == ["text"]
    assert tool.parameters["properties"]["text"] == {"type": "string", "description": "The text to analyze."}
    assert tool.invoke({"text": "b a b"}) == "b: 2\na: 1"  # callable from Python too
    assert type(tool.__pyox_native_tool__).__name__ == "PyCapsule"


async def test_agent_runs_rust_tools(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(
            tool_call("count_words", text="to be or not to be"),
            tool_call("most_common_words", text="to be or not to be", top=2),
            tool_call("count_words", words="wrong argument"),
        ),
        text_response("Done"),
    )
    agent = make_agent(tools=[fast_tools.count_words, fast_tools.most_common_words])

    events = await collect_events(agent.execute("Analyze"))

    finished = events_of_type(events, ToolFinished)
    assert [(event.result, event.is_error) for event in finished[:2]] == [("6", False), ("be: 2\nto: 2", False)]
    # The engine calls the Rust function directly: a Python call would have
    # wrapped the message in an exception name ("RuntimeError: ...").
    assert finished[2].is_error
    assert finished[2].result.startswith("Invalid arguments: missing field `text`")
    assert fake_openai.requests[0].body["tools"][0]["function"]["description"] == "Count the words in a text."


async def test_file_tools(fake_openai: FakeOpenAI, make_agent: MakeAgent, tmp_path: Path) -> None:
    path = str(tmp_path / "notes" / "todo.txt")
    fake_openai.queue(
        tool_calls_response(tool_call("write_file", path=path, content="buy milk")),
        tool_calls_response(tool_call("read_file", path=path), tool_call("read_file", path=path + ".nope")),
        text_response("Done"),
    )
    agent = make_agent(tools=[fast_tools.write_file, fast_tools.read_file])

    events = await collect_events(agent.execute("Save a note"))

    finished = events_of_type(events, ToolFinished)
    assert [event.result for event in finished[:2]] == [f"Wrote 8 bytes to {path}", "buy milk"]
    assert finished[2].is_error
    assert finished[2].result.startswith("Could not read")
