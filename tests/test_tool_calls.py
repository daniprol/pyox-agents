"""Tool calls inside a run: Python tools, Rust tools and tool errors."""

from pathlib import Path

import pytest

from conftest import MakeAgent
from factories import collect_events, events_of_type, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import BuiltinTool, Tool, ToolFinished, ToolStarted


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def shout(text: str) -> str:
    """Uppercase text."""
    return text.upper()


async def test_runs_sync_and_async_python_tools(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("add", a=2, b=3), tool_call("shout", text="hi")),
        text_response("Done"),
    )

    events = await collect_events(make_agent(tools=[add, shout]).execute("Go"))

    started = events_of_type(events, ToolStarted)
    assert [(event.call_id, event.name, event.arguments) for event in started] == [
        ("call_0", "add", {"a": 2, "b": 3}),
        ("call_1", "shout", {"text": "hi"}),
    ]
    finished = events_of_type(events, ToolFinished)
    assert [(event.call_id, event.result, event.is_error) for event in finished] == [
        ("call_0", "5", False),
        ("call_1", "HI", False),
    ]


async def test_tool_schemas_and_results_are_sent_to_the_model(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("add", a=2, b=3)), text_response("5"))

    result = await make_agent(tools=[add]).run("Add 2 and 3")

    assert result.turns == 2
    tool_schema = fake_openai.requests[0].body["tools"][0]
    assert tool_schema["function"]["name"] == "add"
    assert tool_schema["function"]["parameters"]["required"] == ["a", "b"]
    follow_up_messages = fake_openai.requests[1].body["messages"]
    assert follow_up_messages[1]["tool_calls"][0]["function"] == {"name": "add", "arguments": '{"a":2,"b":3}'}
    assert follow_up_messages[2] == {"role": "tool", "tool_call_id": "call_0", "content": "5"}


async def test_failing_and_unknown_tools_are_reported_to_the_model(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    def explode() -> str:
        raise ValueError("boom")

    fake_openai.queue(tool_calls_response(tool_call("explode"), tool_call("missing")), text_response("Sorry"))

    events = await collect_events(make_agent(tools=[explode]).execute("Go"))

    assert [(event.result, event.is_error) for event in events_of_type(events, ToolFinished)] == [
        ("ValueError: boom", True),
        ('Unknown tool "missing"', True),
    ]


async def test_builtin_rust_tools(fake_openai: FakeOpenAI, make_agent: MakeAgent, tmp_path: Path) -> None:
    path = str(tmp_path / "notes" / "todo.txt")
    fake_openai.queue(
        tool_calls_response(tool_call("write_file", path=path, content="buy milk")),
        tool_calls_response(tool_call("read_file", path=path)),
        text_response("Done"),
    )
    agent = make_agent(tools=[BuiltinTool("write_file"), BuiltinTool("read_file")])

    events = await collect_events(agent.execute("Save a note"))

    assert [event.result for event in events_of_type(events, ToolFinished)] == [f"Wrote 8 bytes to {path}", "buy milk"]


async def test_builtin_tool_errors_are_reported(fake_openai: FakeOpenAI, make_agent: MakeAgent, tmp_path: Path) -> None:
    fake_openai.queue(tool_calls_response(tool_call("read_file", path=str(tmp_path / "nope.txt"))), text_response("."))

    events = await collect_events(make_agent(tools=[BuiltinTool("read_file")]).execute("Read"))

    [finished] = events_of_type(events, ToolFinished)
    assert finished.is_error
    assert finished.result.startswith("Could not read")


async def test_threaded_tool_like_a_compiled_rust_function(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    # A builtin (C-implemented) function stands in for a function from a compiled Rust package:
    # no Python signature to inspect, so the schema is given explicitly.
    word_lengths = Tool(
        name="word_lengths",
        description="Length of each word.",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        function=lambda text: list(map(len, text.split())),
        run_in_thread=True,
    )
    fake_openai.queue(tool_calls_response(tool_call("word_lengths", text="to be or")), text_response("."))

    events = await collect_events(make_agent(tools=[word_lengths]).execute("Count"))

    assert [event.result for event in events_of_type(events, ToolFinished)] == ["[2, 2, 2]"]


def test_builtin_tool_lookup() -> None:
    assert BuiltinTool.available() == ["read_file", "write_file"]
    assert BuiltinTool("read_file").name == "read_file"
    with pytest.raises(ValueError, match="Unknown built-in tool"):
        BuiltinTool("nope")


class Greeter:
    """Any object with name, description, parameters and a sync or async `invoke` is a tool."""

    def __init__(self) -> None:
        self.name = "greet"
        self.description = "Greet someone."
        self.parameters = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

    def invoke(self, arguments: dict[str, str]) -> str:
        return f"Hello {arguments['name']}"


async def test_custom_tool_objects_with_sync_invoke(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("greet", name="Dani")), text_response("."))

    events = await collect_events(make_agent(tools=[Greeter()]).execute("Greet Dani"))

    assert [event.result for event in events_of_type(events, ToolFinished)] == ["Hello Dani"]
