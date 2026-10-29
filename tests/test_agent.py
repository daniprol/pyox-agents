"""Running agents: streaming, requests sent, history and concurrency."""

import asyncio

from conftest import MakeAgent
from factories import (
    collect_events,
    events_of_type,
    text_pieces_response,
    text_response,
    tool_call,
    tool_calls_response,
)
from fake_openai import FakeOpenAI
from pyox_agents import RunFinished, TextDelta, Usage


async def test_streams_text_then_finishes(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hello there!"))

    events = await collect_events(make_agent().execute("Hi"))

    assert [event.text for event in events_of_type(events, TextDelta)] == ["Hello ", "there!"]
    assert isinstance(events[-1], RunFinished)
    assert len(events_of_type(events, RunFinished)) == 1


async def test_run_finished_has_text_usage_and_messages(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hello!", usage={"prompt_tokens": 7, "completion_tokens": 3}))

    result = await make_agent().run("Hi")

    assert (result.text, result.turns) == ("Hello!", 1)
    assert result.usage == Usage(input_tokens=7, output_tokens=3, requests=1)
    assert result.messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "tool_calls": [], "model": "test-model"},
    ]


async def test_request_contains_model_prompt_parameters_and_headers(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    fake_openai.queue(text_response("ok"))
    agent = make_agent(prompt="Be nice.", model_parameters={"temperature": 0, "stream": False})

    await agent.run("Hi")

    request = fake_openai.requests[0]
    assert request.body["model"] == "test-model"
    assert request.body["temperature"] == 0
    assert request.body["stream"] is True  # library-controlled fields win over model_parameters
    assert request.body["messages"] == [{"role": "system", "content": "Be nice."}, {"role": "user", "content": "Hi"}]
    assert "tools" not in request.body
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-test"] == "yes"


async def test_history_continues_a_conversation(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hi Dani"), text_response("Your name is Dani"))
    agent = make_agent()

    first = await agent.run("I am Dani")
    second = await agent.run("What is my name?", history=first.messages)

    assert second.text == "Your name is Dani"
    sent_contents = [message["content"] for message in fake_openai.requests[1].body["messages"]]
    assert sent_contents == ["I am Dani", "Hi Dani", "What is my name?"]


async def test_history_alone_can_start_a_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Sure"))

    await make_agent().run(history=[{"role": "user", "content": "Help me"}])

    assert fake_openai.requests[0].body["messages"] == [{"role": "user", "content": "Help me"}]


async def test_one_agent_serves_concurrent_runs(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(*(text_response("answer") for _ in range(5)))
    agent = make_agent()

    results = await asyncio.gather(*(agent.run(f"question {number}") for number in range(5)))

    assert [result.text for result in results] == ["answer"] * 5
    prompts = sorted(request.body["messages"][0]["content"] for request in fake_openai.requests)
    assert prompts == [f"question {number}" for number in range(5)]


async def test_stopping_early_does_not_hang(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("A long answer"))

    async for event in make_agent().execute("Hi"):
        assert isinstance(event, TextDelta)
        break


async def test_cached_and_reasoning_tokens_are_reported(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 900},
        "completion_tokens_details": {"reasoning_tokens": 12},
    }
    fake_openai.queue(text_response("Hi", usage=usage))

    result = await make_agent().run("Hi")

    assert result.usage == Usage(
        input_tokens=1000, cached_input_tokens=900, output_tokens=20, reasoning_tokens=12, requests=1
    )


async def test_usage_is_summed_over_the_turns_of_a_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("ping"), usage={"prompt_tokens": 100, "completion_tokens": 10}),
        text_response("Done", usage={"prompt_tokens": 120, "completion_tokens": 5}),
    )

    result = await make_agent(tools=[lambda: "pong"]).run("Ping")

    assert result.usage == Usage(input_tokens=220, output_tokens=15, requests=2)


async def test_missing_usage_counts_as_zero(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_pieces_response("Hi"))  # some OpenAI-compatible servers send no usage

    result = await make_agent().run("Hi")

    assert (result.turns, result.usage) == (1, Usage())


def test_usages_can_be_added() -> None:
    first = Usage(input_tokens=10, cached_input_tokens=4, output_tokens=2, reasoning_tokens=1, requests=1)
    second = Usage(input_tokens=5, output_tokens=3, requests=1)

    assert first + second == Usage(
        input_tokens=15, cached_input_tokens=4, output_tokens=5, reasoning_tokens=1, requests=2
    )
    assert repr(Usage()) == (
        "Usage(input_tokens=0, cached_input_tokens=0, output_tokens=0, reasoning_tokens=0, requests=0)"
    )
