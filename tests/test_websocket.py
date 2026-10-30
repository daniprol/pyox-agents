"""Responses API over WebSocket (`Client(transport="websocket")`): connections, continuation, retries, fallback."""

import asyncio
import dataclasses
from typing import Any

import pytest

from conftest import MakeAgent
from factories import collect_events, error_response, events_of_type, responses_answer, text_response, tool_call
from fake_openai import DroppedConnection, FakeOpenAI, RecordedRequest
from pyox_agents import (
    Agent,
    AuthenticationError,
    Client,
    NetworkError,
    RateLimitError,
    RequestTimeoutError,
    RunFinished,
    TextDelta,
)


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@pytest.fixture
def make_websocket_agent(websocket_client: Client) -> MakeAgent:
    """Responses API agents whose client uses WebSocket: `make_websocket_agent(tools=[...])`."""

    def make(**options: Any) -> Agent:
        return Agent(client=websocket_client, model="test-model", api="responses", **options)

    return make


def connections(requests: list[RecordedRequest]) -> list[int]:
    return [request.connection for request in requests]


# The whole tool loop: the first turn sends everything, the next only the tool result.
FIRST_TURN_INPUT = [{"role": "user", "content": "2 + 3?"}]
WHOLE_HISTORY = [
    *FIRST_TURN_INPUT,
    {"type": "function_call", "call_id": "call_0", "name": "add", "arguments": '{"a":2,"b":3}'},
    {"type": "function_call_output", "call_id": "call_0", "output": "5"},
]
TOOL_RESULT_ONLY = WHOLE_HISTORY[-1:]


# Requests and answers -----------------------------------------------------------------------------


async def test_streams_text_over_a_websocket(fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent) -> None:
    fake_openai.queue(responses_answer("Hello there!", response_id="resp_7"))

    events = await collect_events(make_websocket_agent(prompt="Be nice.").execute("Hi"))

    assert [event.text for event in events_of_type(events, TextDelta)] == ["Hello ", "there!"]
    result = events[-1]
    assert isinstance(result, RunFinished)
    assert (result.text, result.response_id) == ("Hello there!", "resp_7")
    request = fake_openai.requests[0]
    assert (request.transport, request.path) == ("websocket", "/v1/responses")
    # The HTTP request body, with a `type` and without `stream`.
    assert request.body == {
        "type": "response.create",
        "model": "test-model",
        "instructions": "Be nice.",
        "input": [{"role": "user", "content": "Hi"}],
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-test"] == "yes"  # the client's custom headers


async def test_a_tool_loop_stays_on_one_connection_and_sends_only_new_items(
    fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent
) -> None:
    fake_openai.queue(
        responses_answer(tool_calls=(tool_call("add", a=2, b=3),), response_id="resp_1"),
        responses_answer("It is 5", response_id="resp_2"),
    )

    result = await make_websocket_agent(tools=[add]).run("2 + 3?")

    assert result.text == "It is 5"
    first_turn, second_turn = (request.body for request in fake_openai.requests)
    assert connections(fake_openai.requests) == [1, 1]
    assert "previous_response_id" not in first_turn
    assert first_turn["input"] == FIRST_TURN_INPUT
    # Even without `store`: the connection's cache has the previous response.
    assert (second_turn["previous_response_id"], second_turn["store"]) == ("resp_1", False)
    assert second_turn["input"] == TOOL_RESULT_ONLY
    assert second_turn["tools"] == first_turn["tools"]  # instructions and tools are sent on every turn


async def test_follows_a_redirect_to_a_websocket_address(
    fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent
) -> None:
    fake_openai.websocket_redirect = True
    fake_openai.queue(responses_answer("Hi"))

    finished = await make_websocket_agent().run("Hello")

    assert finished.text == "Hi"
    assert len(fake_openai.websocket_upgrades) == 2
    assert fake_openai.requests[0].transport == "websocket"


async def test_runs_one_after_another_reuse_the_connection(
    fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent
) -> None:
    fake_openai.queue(responses_answer("Hi Dani", response_id="resp_1"), responses_answer("Dani"))
    agent = make_websocket_agent()

    first = await agent.run("I am Dani")
    await agent.run("My name?", history=first.messages)

    assert connections(fake_openai.requests) == [1, 1]
    assert len(fake_openai.websocket_upgrades) == 1
    # A new run sends its whole input: the connection's cache may have lost the response while idle.
    next_run = fake_openai.requests[1].body
    assert "previous_response_id" not in next_run
    assert [item["content"] for item in next_run["input"]] == ["I am Dani", "Hi Dani", "My name?"]


async def test_concurrent_runs_use_separate_connections(
    fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent
) -> None:
    slow_answer = dataclasses.replace(responses_answer("Hi"), delay_seconds=0.3)
    fake_openai.queue(slow_answer, slow_answer, responses_answer("Hi"))
    agent = make_websocket_agent()

    await asyncio.gather(agent.run("one"), agent.run("two"))
    await agent.run("three")

    assert sorted(connections(fake_openai.requests[:2])) == [1, 2]
    assert connections(fake_openai.requests)[2] in {1, 2}  # both went back to the pool


async def test_stopping_a_run_mid_stream_closes_its_connection(
    fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent
) -> None:
    slow_answer = dataclasses.replace(responses_answer("Hello there!"), chunk_delay_seconds=0.2)
    fake_openai.queue(slow_answer, responses_answer("Hi"))
    agent = make_websocket_agent()

    async for event in agent.execute("Hi"):
        if isinstance(event, TextDelta):
            break  # drops the run, which closes its connection
    await agent.run("Again")

    assert connections(fake_openai.requests) == [1, 2]
    for _ in range(100):
        if 1 in fake_openai.closed_websockets:
            break
        await asyncio.sleep(0.01)
    assert fake_openai.closed_websockets == [1]


# Failures and retries -----------------------------------------------------------------------------


async def test_a_connection_closed_before_any_event_is_retried_on_a_new_one(
    fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent
) -> None:
    # The second run gets the pooled connection, which the server closes, as it may do with idle ones.
    fake_openai.queue(responses_answer("one"), DroppedConnection(), responses_answer("two"))
    agent = make_websocket_agent()

    await agent.run("Hi")
    result = await agent.run("Hi again")

    assert result.text == "two"
    assert connections(fake_openai.requests) == [1, 1, 2]
    assert fake_openai.requests[2].body == fake_openai.requests[1].body


async def test_a_connection_closed_mid_stream_raises(fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent) -> None:
    started = responses_answer("Hello there!")
    cut = dataclasses.replace(started, chunks=started.chunks[:2], close_after_chunks=True)  # no `response.completed`
    fake_openai.queue(cut, responses_answer("Hi"))
    agent = make_websocket_agent()

    # Text may have reached the user, so the request is not sent again.
    with pytest.raises(NetworkError):
        await agent.run("Hi")
    await agent.run("Again")

    assert connections(fake_openai.requests) == [1, 2]


async def test_connection_limit_reconnects_and_sends_the_whole_input(
    fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent
) -> None:
    fake_openai.queue(
        responses_answer(tool_calls=(tool_call("add", a=2, b=3),), response_id="resp_1"),
        error_response(400, "Connection lasted 60 minutes", code="websocket_connection_limit_reached"),
        responses_answer("It is 5", response_id="resp_2"),
    )

    result = await make_websocket_agent(tools=[add]).run("2 + 3?")

    assert result.text == "It is 5"
    assert connections(fake_openai.requests) == [1, 1, 2]
    retry = fake_openai.requests[2].body
    assert "previous_response_id" not in retry  # the new connection has no cache yet
    assert retry["input"] == WHOLE_HISTORY


async def test_a_lost_cached_response_is_replaced_by_the_whole_history(
    fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent
) -> None:
    fake_openai.queue(
        responses_answer(tool_calls=(tool_call("add", a=2, b=3),), response_id="resp_1"),
        error_response(400, "Previous response not found.", code="previous_response_not_found"),
        responses_answer("It is 5", response_id="resp_2"),
    )

    result = await make_websocket_agent(tools=[add]).run("2 + 3?")

    assert result.text == "It is 5"
    assert connections(fake_openai.requests) == [1, 1, 1]  # the connection itself is fine
    assert fake_openai.requests[1].body["input"] == TOOL_RESULT_ONLY
    retry = fake_openai.requests[2].body
    assert "previous_response_id" not in retry
    assert retry["input"] == WHOLE_HISTORY


async def test_api_errors_raise_the_usual_exceptions(fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent) -> None:
    fake_openai.queue(error_response(429, "Slow down", code="rate_limit_exceeded"))

    with pytest.raises(RateLimitError, match="Slow down"):
        await make_websocket_agent().run("Hi")
    assert len(fake_openai.requests) == 1


async def test_slow_answers_time_out(fake_openai: FakeOpenAI) -> None:
    fake_openai.queue(dataclasses.replace(responses_answer("Hi"), delay_seconds=1))
    client = Client(api_key="k", base_url=fake_openai.base_url, request_timeout_seconds=0.2, transport="websocket")

    with pytest.raises(RequestTimeoutError):
        await Agent(client=client, model="m", api="responses").run("Hi")


# Fallback to HTTP ---------------------------------------------------------------------------------


@pytest.mark.parametrize("status", [404, 426])
async def test_a_server_without_websocket_mode_gets_http_requests(
    fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent, status: int
) -> None:
    fake_openai.websocket_refusal_status = status
    fake_openai.queue(responses_answer("one"), responses_answer("two"))
    agent = make_websocket_agent()

    assert (await agent.run("Hi")).text == "one"
    assert (await agent.run("Hi")).text == "two"

    assert [request.transport for request in fake_openai.requests] == ["http", "http"]
    assert len(fake_openai.websocket_upgrades) == 1  # the client does not ask again


async def test_a_refused_key_raises_instead_of_falling_back(
    fake_openai: FakeOpenAI, make_websocket_agent: MakeAgent
) -> None:
    fake_openai.websocket_refusal_status = 401

    with pytest.raises(AuthenticationError):
        await make_websocket_agent().run("Hi")
    assert fake_openai.requests == []


async def test_chat_completions_always_use_http(fake_openai: FakeOpenAI, websocket_client: Client) -> None:
    fake_openai.queue(text_response("Hi"))

    await Agent(client=websocket_client, model="m", api="chat").run("Hi")

    assert fake_openai.requests[0].transport == "http"
    assert fake_openai.websocket_upgrades == []


# Client options -----------------------------------------------------------------------------------


async def test_azure_client_connects_to_its_v1_responses_url(fake_openai: FakeOpenAI) -> None:
    endpoint = fake_openai.base_url.removesuffix("/v1")  # e.g. https://my-resource.openai.azure.com
    fake_openai.queue(responses_answer("Hi"))
    client = Client.azure(endpoint, api_key="azure-key", transport="websocket")

    await Agent(client=client, model="my-deployment", api="responses").run("Hi")

    request = fake_openai.requests[0]
    assert (request.transport, request.path) == ("websocket", "/openai/v1/responses")
    assert request.headers["authorization"] == "Bearer azure-key"


def test_unknown_transport_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown transport"):
        Client(api_key="k", transport="grpc")  # ty: ignore[invalid-argument-type]
