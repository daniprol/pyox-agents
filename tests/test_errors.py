"""Failures surface as typed exceptions from the event stream."""

import socket

import pytest

from conftest import MakeAgent
from factories import error_response, text_response, tool_call, tool_calls_response
from fake_openai import DroppedConnection, FakeOpenAI, StreamResponse
from pyox_agents import (
    Agent,
    AgentError,
    ApiError,
    AuthenticationError,
    BuiltinTool,
    Client,
    InvalidResponseError,
    MaxTurnsExceededError,
    NetworkError,
    RateLimitError,
    RequestTimeoutError,
    Usage,
)


@pytest.mark.parametrize(
    ("status", "exception"),
    [(401, AuthenticationError), (403, AuthenticationError), (429, RateLimitError), (500, ApiError)],
)
async def test_http_errors_raise_typed_exceptions(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, status: int, exception: type[Exception]
) -> None:
    fake_openai.queue(error_response(status, "nope"))

    with pytest.raises(exception, match="nope"):
        await make_agent().run("Hi")


async def test_all_errors_share_a_base_class(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(error_response(400, "bad request"))

    with pytest.raises(AgentError):
        await make_agent().run("Hi")


async def test_error_inside_the_stream(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(StreamResponse([{"error": {"message": "overloaded"}}]))

    with pytest.raises(ApiError, match="overloaded"):
        await make_agent().run("Hi")


async def test_unparseable_chunk(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(StreamResponse([{"choices": "not a list"}]))

    with pytest.raises(InvalidResponseError):
        await make_agent().run("Hi")


async def test_max_turns(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(*(tool_calls_response(tool_call("ping")) for _ in range(2)))

    with pytest.raises(MaxTurnsExceededError) as error:
        await make_agent(tools=[lambda: "pong"], max_turns=2).run("Loop forever")

    # The tokens of a failed run were still used (and billed).
    assert error.value.usage == Usage(input_tokens=20, output_tokens=10, requests=2)


def test_errors_raised_outside_a_run_have_no_usage() -> None:
    assert AgentError("not from a run").usage is None


async def test_request_timeout(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue(StreamResponse(text_response("late").chunks, delay_seconds=1))
    client = Client(api_key="x", base_url=fake_openai.base_url, request_timeout_seconds=0.1)

    with pytest.raises(RequestTimeoutError):
        await Agent(client=client, model="m").run("Hi")


async def test_unreachable_server() -> None:
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    client = Client(api_key="x", base_url=f"http://127.0.0.1:{port}/v1")

    with pytest.raises(NetworkError):
        await Agent(client=client, model="m").run("Hi")


def test_invalid_configuration(client: Client) -> None:
    with pytest.raises(ValueError, match="not a tool"):
        Agent(model="m", client=client, tools=[42])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Two tools"):
        Agent(model="m", client=client, tools=[BuiltinTool("read_file"), BuiltinTool("read_file")])
    with pytest.raises(ValueError, match="Invalid history"):
        Agent(model="m", client=client).execute(history=[{"role": "robot"}])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="prompt"):
        Agent(model="m", client=client).execute()
    with pytest.raises(ValueError, match="header"):
        Client(headers={"bad header": "x"})


async def test_retries_requests_dropped_before_any_response(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue(DroppedConnection(), DroppedConnection(), text_response("Made it"))
    client = Client(api_key="x", base_url=fake_openai.base_url, max_retries=2)

    result = await Agent(client=client, model="m").run("Hi")

    assert result.text == "Made it"
    assert len(fake_openai.requests) == 3


async def test_gives_up_after_max_retries(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue(DroppedConnection(), DroppedConnection())
    client = Client(api_key="x", base_url=fake_openai.base_url, max_retries=1)

    with pytest.raises(NetworkError):
        await Agent(client=client, model="m").run("Hi")
    assert len(fake_openai.requests) == 2
