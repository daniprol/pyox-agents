"""Failures surface as typed exceptions from the event stream."""

import asyncio
import pickle
import socket
import time

import fast_tools
import pytest

from conftest import MakeAgent
from factories import error_response, text_response, tool_call, tool_calls_response
from fake_openai import DroppedConnection, FakeOpenAI, StreamResponse
from pyox_agents import (
    Agent,
    AgentError,
    ApiError,
    AuthenticationError,
    Client,
    HookError,
    Hooks,
    InputRequested,
    InvalidResponseError,
    MaxTurnsExceededError,
    NetworkError,
    RateLimitError,
    RequestTimeoutError,
    RunContext,
    RunFinished,
    Usage,
    tool,
)


@pytest.mark.parametrize(
    ("status", "exception"),
    [(401, AuthenticationError), (403, AuthenticationError), (429, RateLimitError), (500, ApiError)],
)
async def test_http_errors_raise_typed_exceptions(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, status: int, exception: type[Exception]
) -> None:
    # 429 and 5xx are retried (twice by default) before they raise.
    fake_openai.queue(*[error_response(status, "nope", headers={"retry-after-ms": "1"})] * 3)

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
        Agent(model="m", client=client, tools=[fast_tools.count_words, fast_tools.count_words])
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


# Retries ------------------------------------------------------------------------------------------


async def test_rate_limits_are_retried_after_the_wait_the_server_asks(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    fake_openai.queue(error_response(429, "Slow down", headers={"retry-after-ms": "10"}), text_response("Hi"))

    result = await make_agent().run("Hi")

    assert result.text == "Hi"
    assert len(fake_openai.requests) == 2


async def test_server_errors_raise_with_their_details_after_the_retries(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    headers = {"retry-after-ms": "1", "x-request-id": "req_123"}
    fake_openai.queue(*[error_response(503, "Overloaded", headers=headers)] * 3)

    with pytest.raises(ApiError) as error:
        await make_agent().run("Hi")

    assert len(fake_openai.requests) == 3  # the request and 2 retries (`Client(max_retries=2)`)
    assert error.value.status_code == 503
    assert error.value.body == {"error": {"message": "Overloaded"}}
    assert error.value.request_id == "req_123"
    assert error.value.retry_after_seconds == 0.001
    assert error.value.usage == Usage()  # no request got an answer


async def test_a_long_retry_after_raises_at_once(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(error_response(429, "Quota", headers={"retry-after": "3600"}), text_response("Hi"))

    started = time.monotonic()
    with pytest.raises(RateLimitError) as error:
        await make_agent().run("Hi")

    assert time.monotonic() - started < 1
    assert error.value.retry_after_seconds == 3600
    assert len(fake_openai.requests) == 1


async def test_client_errors_are_not_retried(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.queue(error_response(400, "Bad request"), text_response("Hi"))

    with pytest.raises(ApiError, match="Bad request") as error:
        await Agent(client=client, model="m").run("Hi")

    assert error.value.status_code == 400
    assert len(fake_openai.requests) == 1


def test_api_error_fields_are_none_outside_the_api() -> None:
    error = ApiError("made by hand")
    assert (error.status_code, error.body, error.request_id, error.retry_after_seconds) == (None, None, None, None)


async def test_api_errors_have_a_short_message_and_a_code(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(error_response(400, "Too long", code="context_length_exceeded"))

    with pytest.raises(ApiError) as error:
        await make_agent().run("Hi")

    assert str(error.value) == "HTTP 400: Too long"  # the whole body is in `error.body`
    assert error.value.code == "context_length_exceeded"


async def test_rate_limits_inside_the_stream_keep_the_request_id(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    error_chunk = {"error": {"message": "Slow down", "code": "rate_limit_exceeded"}}
    fake_openai.queue(StreamResponse([error_chunk], headers={"x-request-id": "req_9"}))

    with pytest.raises(RateLimitError, match=r"^Slow down$") as error:
        await make_agent().run("Hi")

    assert (error.value.status_code, error.value.code, error.value.request_id) == (None, "rate_limit_exceeded", "req_9")


async def test_network_errors_say_why(make_agent: MakeAgent) -> None:
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    client = Client(api_key="x", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0)

    with pytest.raises(NetworkError, match=r"(?i)connection refused"):
        await Agent(client=client, model="m").run("Hi")


@pytest.mark.parametrize("retry_after", ["inf", "1e999", "nan"])
async def test_odd_retry_after_headers_are_ignored(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, retry_after: str
) -> None:
    fake_openai.queue(error_response(400, "Bad", headers={"retry-after": retry_after}))

    with pytest.raises(ApiError) as error:
        await make_agent().run("Hi")

    assert error.value.retry_after_seconds is None


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf")])
def test_timeouts_must_be_positive(seconds: float) -> None:
    with pytest.raises(ValueError, match="must be a positive number of seconds"):
        Client(connect_timeout_seconds=seconds)
    with pytest.raises(ValueError, match="must be a positive number of seconds"):
        Client(request_timeout_seconds=seconds)


async def test_run_errors_can_be_pickled(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    # Celery, multiprocessing and Ray send exceptions between processes with pickle.
    fake_openai.queue(text_response("Hi"), error_response(400, "Bad", code="bad"))
    agent = make_agent()
    result = await agent.run("Hi")

    with pytest.raises(ApiError) as error:
        await agent.run("Again", history=result.messages)

    copy = pickle.loads(pickle.dumps(error.value))
    assert (copy.status_code, copy.code, copy.usage, copy.messages) == (400, "bad", Usage(), error.value.messages)


# Runs that fail at the end keep their work ------------------------------------------------------


async def test_a_failing_after_run_hook_keeps_the_finished_conversation(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    def save_somewhere(ctx: RunContext, result: RunFinished) -> None:
        raise ConnectionError("database is down")

    fake_openai.queue(text_response("Hello!"))

    with pytest.raises(HookError) as error:
        await make_agent(hooks=[Hooks(after_run=save_somewhere)]).run("Hi")

    assert error.value.messages is not None
    assert [message["role"] for message in error.value.messages] == ["user", "assistant"]


async def test_on_request_errors_inside_ctx_ask_end_the_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    async def ask_name(ctx: RunContext) -> str:
        """Ask the user's name."""
        answer = await ctx.ask("What is your name?")
        return answer.value

    def broken_handler(request: InputRequested) -> None:
        raise RuntimeError("the UI is gone")

    fake_openai.queue(tool_calls_response(tool_call("ask_name")))

    with pytest.raises(HookError, match="on_request handler raised RuntimeError") as error:
        await make_agent(tools=[ask_name]).run("Hi", on_request=broken_handler)
    assert isinstance(error.value.__cause__, RuntimeError)


async def test_stopping_a_run_cancels_its_async_tools(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()

    @tool
    async def slow() -> str:
        """Takes a while."""
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "done"

    fake_openai.queue(tool_calls_response(tool_call("slow")))
    agent = make_agent(tools=[slow])

    async def read_events() -> None:
        async for _ in agent.execute("Go"):
            pass

    # Cancelling the task reading the run drops the run (a web request cancelled, say).
    reading = asyncio.create_task(read_events())
    await asyncio.wait_for(started.wait(), 1)
    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading

    await asyncio.wait_for(cancelled.wait(), 1)  # the tool did not go on alone
