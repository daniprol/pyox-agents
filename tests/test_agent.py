import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fake_openai import FakeOpenAI, text_response, tool_calls_response

from pyox_agents import (
    Agent,
    AuthenticationError,
    BuiltinTool,
    Client,
    MaxTurnsExceededError,
    RateLimitError,
    RunFinished,
    TextDelta,
    ToolFinished,
    ToolStarted,
)


async def collect(agent: Agent, prompt: str, **kwargs: Any) -> list[Any]:
    return [event async for event in agent.execute(prompt, **kwargs)]


def test_streams_text_and_finishes(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.responses.append(text_response("Hello there!"))
    agent = Agent(client=client, model="test-model", prompt="Be nice.", model_parameters={"temperature": 0})

    events = asyncio.run(collect(agent, "Hi"))

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["Hello ", "there!"]
    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert (finished.text, finished.turns, finished.input_tokens, finished.output_tokens) == ("Hello there!", 1, 10, 5)
    assert finished.messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there!", "tool_calls": []},
    ]

    request = fake_openai.requests[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0
    assert request["stream"] is True
    assert request["messages"] == [{"role": "system", "content": "Be nice."}, {"role": "user", "content": "Hi"}]
    assert fake_openai.headers[0]["authorization"] == "Bearer test-key"
    assert fake_openai.headers[0]["x-test"] == "yes"


def test_runs_python_tools_and_sends_results_back(fake_openai: FakeOpenAI, client: Client) -> None:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async def shout(text: str) -> str:
        """Uppercase text."""
        return text.upper()

    fake_openai.responses += [
        tool_calls_response(("call_1", "add", {"a": 2, "b": 3}), ("call_2", "shout", {"text": "hi"})),
        text_response("2 + 3 = 5"),
    ]
    agent = Agent(client=client, model="m", tools=[add, shout])

    events = asyncio.run(collect(agent, "Add 2 and 3"))

    started = [event for event in events if isinstance(event, ToolStarted)]
    assert [(event.call_id, event.name, event.arguments) for event in started] == [
        ("call_1", "add", {"a": 2, "b": 3}),
        ("call_2", "shout", {"text": "hi"}),
    ]
    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert [(event.result, event.is_error) for event in finished] == [("5", False), ("HI", False)]
    assert events[-1].turns == 2

    tool_schemas = fake_openai.requests[0]["tools"]
    assert tool_schemas[0]["function"]["name"] == "add"
    assert tool_schemas[0]["function"]["parameters"]["required"] == ["a", "b"]

    second_request_messages = fake_openai.requests[1]["messages"]
    assert second_request_messages[1]["tool_calls"][0]["function"] == {"name": "add", "arguments": '{"a":2,"b":3}'}
    assert second_request_messages[2] == {"role": "tool", "tool_call_id": "call_1", "content": "5"}
    assert second_request_messages[3] == {"role": "tool", "tool_call_id": "call_2", "content": "HI"}


def test_tool_errors_are_reported_to_the_model(fake_openai: FakeOpenAI, client: Client) -> None:
    def explode() -> str:
        raise ValueError("boom")

    fake_openai.responses += [
        tool_calls_response(("c1", "explode", {}), ("c2", "missing", {})),
        text_response("Sorry"),
    ]
    agent = Agent(client=client, model="m", tools=[explode])

    events = asyncio.run(collect(agent, "Go"))

    finished = [event for event in events if isinstance(event, ToolFinished)]
    assert [(event.result, event.is_error) for event in finished] == [
        ("ValueError: boom", True),
        ('Unknown tool "missing"', True),
    ]


def test_builtin_rust_tools(fake_openai: FakeOpenAI, client: Client, tmp_path: Path) -> None:
    path = str(tmp_path / "notes" / "todo.txt")
    fake_openai.responses += [
        tool_calls_response(("c1", "write_file", {"path": path, "content": "buy milk"})),
        tool_calls_response(("c2", "read_file", {"path": path})),
        text_response("Done"),
    ]
    agent = Agent(client=client, model="m", tools=[BuiltinTool("write_file"), BuiltinTool("read_file")])

    events = asyncio.run(collect(agent, "Save a note"))

    results = [event.result for event in events if isinstance(event, ToolFinished)]
    assert results == [f"Wrote 8 bytes to {path}", "buy milk"]
    assert "read_file" in BuiltinTool.available()
    with pytest.raises(ValueError, match="Unknown built-in tool"):
        BuiltinTool("nope")


def test_history_continues_a_conversation(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.responses += [text_response("Hi Dani"), text_response("Your name is Dani")]
    agent = Agent(client=client, model="m")

    first = asyncio.run(agent.run("I am Dani"))
    second = asyncio.run(agent.run("What is my name?", history=first.messages))

    assert second.text == "Your name is Dani"
    assert [message["content"] for message in fake_openai.requests[1]["messages"]] == [
        "I am Dani",
        "Hi Dani",
        "What is my name?",
    ]


def test_one_agent_serves_concurrent_runs(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.responses += [text_response("same answer") for _ in range(5)]
    agent = Agent(client=client, model="m")

    async def run_five() -> list[RunFinished]:
        return await asyncio.gather(*(agent.run(f"question {number}") for number in range(5)))

    results = asyncio.run(run_five())

    assert [result.text for result in results] == ["same answer"] * 5
    assert sorted(request["messages"][0]["content"] for request in fake_openai.requests) == [
        f"question {number}" for number in range(5)
    ]


@pytest.mark.parametrize(
    ("status", "exception"),
    [(401, AuthenticationError), (429, RateLimitError)],
)
def test_http_errors_raise_typed_exceptions(
    fake_openai: FakeOpenAI, client: Client, status: int, exception: type[Exception]
) -> None:
    fake_openai.responses.append((status, json.dumps({"error": {"message": "nope"}})))
    agent = Agent(client=client, model="m")

    with pytest.raises(exception, match="nope"):
        asyncio.run(agent.run("Hi"))


def test_max_turns(fake_openai: FakeOpenAI, client: Client) -> None:
    def ping() -> str:
        return "pong"

    fake_openai.responses += [tool_calls_response((f"c{turn}", "ping", {})) for turn in range(2)]
    agent = Agent(client=client, model="m", tools=[ping], max_turns=2)

    with pytest.raises(MaxTurnsExceededError):
        asyncio.run(agent.run("Loop forever"))


def test_invalid_configuration() -> None:
    client = Client(api_key="x")
    with pytest.raises(ValueError, match="not a tool"):
        Agent(model="m", client=client, tools=[42])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Two tools"):
        Agent(model="m", client=client, tools=[BuiltinTool("read_file"), BuiltinTool("read_file")])
    with pytest.raises(ValueError, match="Invalid history"):
        Agent(model="m", client=client).execute(history=[{"role": "robot"}])


def test_stopping_early_cancels_the_run(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.responses.append(text_response("A long answer"))
    agent = Agent(client=client, model="m")

    async def first_event() -> Any:
        async for event in agent.execute("Hi"):
            return event

    assert isinstance(asyncio.run(first_event()), TextDelta)
