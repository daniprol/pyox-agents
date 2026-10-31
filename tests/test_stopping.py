"""How runs end: `RunFinished.stop_reason`, `run.stop()`, and errors that keep the conversation."""

import asyncio
import dataclasses
import json
from typing import Any

import pytest

from conftest import MakeAgent
from factories import (
    collect_events,
    error_response,
    events_of_type,
    responses_answer,
    run_finished,
    text_response,
    tool_call,
    tool_calls_response,
)
from fake_openai import FakeOpenAI, StreamResponse
from pyox_agents import (
    ApiError,
    Client,
    MaxTurnsExceededError,
    Message,
    TextDelta,
    ToolFinished,
    ToolStarted,
    tool,
)

ran: list[str] = []


@tool
def ping() -> str:
    """Ping."""
    ran.append("ping")
    return "pong"


def cut_by_the_token_limit(*pieces: str, with_a_tool_call: bool = False) -> StreamResponse:
    """A Chat Completions answer that ends with `finish_reason: "length"`, maybe in the middle of a tool call."""
    chunks: list[dict[str, Any]] = [{"choices": [{"delta": {"content": piece}}]} for piece in pieces]
    if with_a_tool_call:
        half_call = {"index": 0, "id": "call_0", "function": {"name": "ping", "arguments": '{"a": '}}
        chunks.append({"choices": [{"delta": {"tool_calls": [half_call]}}]})
    chunks.append({"choices": [{"delta": {}, "finish_reason": "length"}]})
    return StreamResponse(chunks)


def slow_text(*pieces: str) -> StreamResponse:
    """An answer whose pieces arrive 0.1 s apart, so a run can be stopped in the middle."""
    chunks: list[dict[str, Any]] = [{"choices": [{"delta": {"content": piece}}]} for piece in pieces]
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    return StreamResponse(chunks, chunk_delay_seconds=0.1)


async def test_a_normal_answer_ends_with_end(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hi"))

    assert (await make_agent().run("Hi")).stop_reason == "end"


async def test_an_answer_cut_by_the_token_limit_ends_with_length(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    fake_openai.queue(cut_by_the_token_limit("Once upon a", with_a_tool_call=True))
    ran.clear()

    result = await make_agent(tools=[ping]).run("Tell me a story")

    assert (result.stop_reason, result.text) == ("length", "Once upon a")
    assert ran == []  # a cut tool call never runs
    assert result.messages[-1] == {
        "role": "assistant",
        "content": "Once upon a",
        "tool_calls": [],
        "model": "test-model",
    }
    assert len(fake_openai.requests) == 1


async def test_a_refusal_ends_with_refusal(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        StreamResponse(
            [
                {"choices": [{"delta": {"refusal": "I can't help with that."}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
        )
    )

    result = await make_agent().run("Something bad")

    assert (result.stop_reason, result.text) == ("refusal", "I can't help with that.")


async def test_an_incomplete_response_ends_with_length(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    answer = responses_answer("Once upon a")
    incomplete = {
        "type": "response.incomplete",
        "response": {**answer.chunks[-1]["response"], "incomplete_details": {"reason": "max_output_tokens"}},
    }
    fake_openai.queue(dataclasses.replace(answer, chunks=[*answer.chunks[:-1], incomplete]))

    result = await make_agent(api="responses").run("Tell me a story")

    assert (result.stop_reason, result.text) == ("length", "Once upon a")


# run.stop() ---------------------------------------------------------------------------------------


@dataclasses.dataclass
class Store:
    saved: list[list[Message]] = dataclasses.field(default_factory=list)

    def load(self, conversation_id: str) -> list[Message]:
        return []

    def save(self, conversation_id: str, new_messages_json: str) -> None:
        self.saved.append(json.loads(new_messages_json))


async def test_stop_keeps_the_partial_answer_and_saves_it(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(slow_text("Once ", "upon ", "a ", "time ", "there ", "was ", "a ", "cat."))
    store = Store()
    run = make_agent(history_store=store).execute("Tell me a story", conversation_id="c1")

    events = []
    async for event in run:
        events.append(event)
        if isinstance(event, TextDelta):
            run.stop()

    result = run_finished(events)
    assert result.stop_reason == "stopped"
    assert result.text == "".join(delta.text for delta in events_of_type(events, TextDelta))
    assert result.text.startswith("Once ")
    assert not result.text.endswith("cat.")
    assert result.messages[-1] == {"role": "assistant", "content": result.text, "tool_calls": [], "model": "test-model"}
    assert store.saved == [result.new_messages]


async def test_stop_lets_running_tools_finish_and_sends_nothing_more(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    stopped = asyncio.Event()

    @tool
    async def ping() -> str:
        """Ping."""
        await stopped.wait()  # still running when `run.stop()` comes
        return "pong"

    fake_openai.queue(tool_calls_response(tool_call("ping")), text_response("Never sent"))
    run = make_agent(tools=[ping]).execute("Ping")

    events = []
    async for event in run:
        events.append(event)
        if isinstance(event, ToolStarted):
            run.stop()
            stopped.set()

    result = run_finished(events)
    assert result.stop_reason == "stopped"
    assert [event.result for event in events_of_type(events, ToolFinished)] == ["pong"]
    assert result.messages[-1]["role"] == "tool"  # a valid history, to continue later
    assert len(fake_openai.requests) == 1


async def test_a_run_stopped_before_it_starts_sends_nothing(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    run = make_agent().execute("Hi")
    run.stop()

    result = run_finished(await collect_events(run))

    assert (result.stop_reason, result.turns, len(fake_openai.requests)) == ("stopped", 0, 0)
    assert result.messages == [{"role": "user", "content": "Hi"}]


async def test_stop_works_over_websocket(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, websocket_client: Client
) -> None:
    answer = responses_answer("Once upon a time")
    pieces = [{"type": "response.output_text.delta", "delta": word} for word in ("Once ", "upon ", "a ", "time")]
    fake_openai.queue(
        dataclasses.replace(answer, chunks=[answer.chunks[0], *pieces, answer.chunks[-1]], chunk_delay_seconds=0.1)
    )
    run = make_agent(api="responses", client=websocket_client).execute("Story")

    events = []
    async for event in run:
        events.append(event)
        if isinstance(event, TextDelta):
            run.stop()

    result = run_finished(events)
    assert (result.stop_reason, result.text) == ("stopped", "Once ")


# Errors keep the conversation ---------------------------------------------------------------------


async def test_max_turns_keeps_the_conversation_to_continue_it(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("ping")))
    agent = make_agent(tools=[ping], max_turns=1)

    with pytest.raises(MaxTurnsExceededError) as error:
        await agent.run("Ping")

    messages = error.value.messages
    assert messages is not None
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]

    fake_openai.queue(text_response("Pong it is"))
    result = await agent.run(history=messages)
    assert result.text == "Pong it is"


async def test_api_errors_keep_the_conversation(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("ping")), error_response(400, "Bad request"))

    with pytest.raises(ApiError) as error:
        await make_agent(tools=[ping]).run("Ping")

    assert error.value.messages is not None
    assert error.value.messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "name": "ping",
        "content": "pong",
        "is_error": False,
    }


def test_errors_raised_outside_a_run_have_no_messages() -> None:
    assert ApiError("by hand").messages is None
