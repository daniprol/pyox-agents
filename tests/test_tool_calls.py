"""Tool calls inside a run: Python tools, Rust tools and tool errors."""

import asyncio
from typing import Any

import pytest

from conftest import MakeAgent
from factories import collect_events, events_of_type, finished_calls, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import Message, Tool, ToolFinished, ToolStarted, tool


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

    events = await collect_events(make_agent(tools=[add, shout]).stream("Go"))

    started = events_of_type(events, ToolStarted)
    assert [(event.call_id, event.name, event.arguments) for event in started] == [
        ("call_0", "add", {"a": 2, "b": 3}),
        ("call_1", "shout", {"text": "hi"}),
    ]
    finished = finished_calls(events)
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


async def test_each_result_streams_as_its_call_ends_and_the_model_reads_them_in_order(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    app_saw_fast = asyncio.Event()

    @tool
    async def slow() -> str:
        """Ends only once the app has the other call's result."""
        await asyncio.wait_for(app_saw_fast.wait(), timeout=5)
        return "slow"

    @tool
    async def fast() -> str:
        """Ends at once."""
        return "fast"

    fake_openai.queue(tool_calls_response(tool_call("slow"), tool_call("fast")), text_response("Done"))

    finished = []
    async for event in make_agent(tools=[slow, fast]).stream("Go"):
        if isinstance(event, ToolFinished):
            finished.append(event.result)
            app_saw_fast.set()

    assert finished == ["fast", "slow"]
    sent = [message for message in fake_openai.requests[1].body["messages"] if message["role"] == "tool"]
    assert [(message["tool_call_id"], message["content"]) for message in sent] == [
        ("call_0", "slow"),
        ("call_1", "fast"),
    ]  # the model reads the results in the order it asked for the calls


async def test_failing_and_unknown_tools_are_reported_to_the_model(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    def explode() -> str:
        raise ValueError("boom")

    fake_openai.queue(tool_calls_response(tool_call("explode"), tool_call("missing")), text_response("Sorry"))

    events = await collect_events(make_agent(tools=[explode]).stream("Go"))

    assert [(event.result, event.is_error) for event in finished_calls(events)] == [
        ("ValueError: boom", True),
        ('Unknown tool "missing"', True),
    ]


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

    events = await collect_events(make_agent(tools=[word_lengths]).stream("Count"))

    assert [event.result for event in events_of_type(events, ToolFinished)] == ["[2, 2, 2]"]


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

    events = await collect_events(make_agent(tools=[Greeter()]).stream("Greet Dani"))

    assert [event.result for event in events_of_type(events, ToolFinished)] == ["Hello Dani"]


# Tools for one run (`tools=`) ------------------------------------------------------------------


def tools_for(user: str) -> list[Tool]:
    """Tools made for one user, as an app would make them for each request."""

    def whoami() -> str:
        """The user's name."""
        return user

    return [tool(whoami)]


def tool_results(messages: list[Message]) -> list[Any]:
    return [message["content"] for message in messages if message["role"] == "tool"]


async def test_each_run_can_have_its_own_tools(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("whoami")),
        text_response("Hi Ana"),
        tool_calls_response(tool_call("whoami")),
        text_response("Hi Ben"),
    )
    agent = make_agent(tools=[add])

    ana = await agent.run("Who am I?", tools=tools_for("ana"))
    ben = await agent.run("Who am I?", tools=tools_for("ben"))

    assert (tool_results(ana.new_messages), tool_results(ben.new_messages)) == (["ana"], ["ben"])
    sent_tools = [schema["function"]["name"] for schema in fake_openai.requests[0].body["tools"]]
    assert sent_tools == ["add", "whoami"]
    assert len(agent.tools) == 1  # the agent itself did not change


async def test_a_run_without_its_tools_does_not_have_them(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hi"), text_response("Hi"))
    agent = make_agent(tools=[add])

    await agent.run("Hi", tools=tools_for("ana"))
    await agent.run("Hi")

    assert [schema["function"]["name"] for schema in fake_openai.requests[1].body["tools"]] == ["add"]


def test_tools_for_a_run_need_names_of_their_own(make_agent: MakeAgent) -> None:
    agent = make_agent(tools=[add])

    with pytest.raises(ValueError, match='"add"'):
        agent.stream("Hi", tools=[add])
    with pytest.raises(ValueError, match='"whoami"'):
        agent.stream("Hi", tools=[*tools_for("ana"), *tools_for("ben")])
    with pytest.raises(ValueError, match='"whoami"'):
        agent.stream("Hi", tools=tools_for("ana"), frontend_tools=[{"name": "whoami"}])
