"""The Responses API (`Agent(api="responses")`): requests, reasoning and stored responses."""

from typing import Any

import pytest

from conftest import MakeAgent
from factories import collect_events, error_response, events_of_type, responses_answer, text_response, tool_call
from fake_openai import ErrorResponse, FakeOpenAI, StreamResponse
from pyox_agents import ApiError, Hooks, InvalidResponseError, Message, RunContext, RunFinished, TextDelta, Usage, image


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@pytest.fixture
def make_responses_agent(make_agent: MakeAgent) -> MakeAgent:
    """Like `make_agent`, for agents using the Responses API: `make_responses_agent(store=True)`."""

    def make(**options: Any) -> Any:
        return make_agent(api="responses", **options)

    return make


# Requests and answers -----------------------------------------------------------------------------


async def test_streams_text_and_sends_a_stateless_request(
    fake_openai: FakeOpenAI, make_responses_agent: MakeAgent
) -> None:
    fake_openai.queue(responses_answer("Hello there!", response_id="resp_7"))

    events = await collect_events(make_responses_agent(prompt="Be nice.").execute("Hi"))

    assert [event.text for event in events_of_type(events, TextDelta)] == ["Hello ", "there!"]
    result = events[-1]
    assert isinstance(result, RunFinished)
    assert (result.text, result.response_id) == ("Hello there!", "resp_7")
    request = fake_openai.requests[0]
    assert request.path == "/v1/responses"
    assert request.body == {
        "model": "test-model",
        "instructions": "Be nice.",
        "input": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }


async def test_tool_calls_and_their_results_become_input_items(
    fake_openai: FakeOpenAI, make_responses_agent: MakeAgent
) -> None:
    fake_openai.queue(responses_answer(tool_calls=(tool_call("add", a=2, b=3),)), responses_answer("It is 5"))

    result = await make_responses_agent(tools=[add]).run("2 + 3?")

    assert result.text == "It is 5"
    first_request, second_request = (request.body for request in fake_openai.requests)
    assert first_request["tools"] == [
        {
            "type": "function",
            "name": "add",
            "description": "Add two numbers.",
            "parameters": first_request["tools"][0]["parameters"],  # the schema itself is tested with tools
            "strict": False,
        }
    ]
    # Without `store`, every request sends the whole conversation.
    assert second_request["input"] == [
        {"role": "user", "content": "2 + 3?"},
        {"type": "function_call", "call_id": "call_0", "name": "add", "arguments": '{"a":2,"b":3}'},
        {"type": "function_call_output", "call_id": "call_0", "output": "5"},
    ]


async def test_prompt_images_become_input_parts(fake_openai: FakeOpenAI, make_responses_agent: MakeAgent) -> None:
    fake_openai.queue(responses_answer("A cat"))

    await make_responses_agent().run(["What is this?", image("https://example.com/cat.png")])

    assert fake_openai.requests[0].body["input"][0]["content"] == [
        {"type": "input_text", "text": "What is this?"},
        {"type": "input_image", "image_url": "https://example.com/cat.png"},
    ]


async def test_usage_is_reported_and_summed_over_turns(
    fake_openai: FakeOpenAI, make_responses_agent: MakeAgent
) -> None:
    first_usage = {
        "input_tokens": 1000,
        "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 900},
        "output_tokens_details": {"reasoning_tokens": 12},
    }
    fake_openai.queue(
        responses_answer(tool_calls=(tool_call("add", a=1, b=2),), usage=first_usage),
        responses_answer("3", response_id="resp_2", usage={"input_tokens": 1050, "output_tokens": 5}),
    )

    result = await make_responses_agent(tools=[add]).run("Add")

    assert result.usage == Usage(
        input_tokens=2050, cached_input_tokens=900, output_tokens=25, reasoning_tokens=12, requests=2
    )


async def test_model_parameters_are_passed_through(fake_openai: FakeOpenAI, make_responses_agent: MakeAgent) -> None:
    fake_openai.queue(responses_answer("Hi"))
    parameters = {"reasoning": {"effort": "low"}, "include": ["web_search_call.results"], "store": True}

    await make_responses_agent(model_parameters=parameters).run("Hi")

    body = fake_openai.requests[0].body
    assert body["reasoning"] == {"effort": "low"}
    assert body["include"] == ["web_search_call.results", "reasoning.encrypted_content"]  # ours is added
    assert body["store"] is False  # library-controlled fields win: use `store=True` on the agent


# Reasoning ------------------------------------------------------------------------------------------


async def test_reasoning_is_kept_and_sent_back_unchanged(
    fake_openai: FakeOpenAI, make_responses_agent: MakeAgent
) -> None:
    fake_openai.queue(
        responses_answer("I am thinking about it", reasoning=("The user greets.",), response_id="resp_1"),
        responses_answer("Fine"),
    )
    agent = make_responses_agent()

    first = await agent.run("Hi")
    await agent.run("And now?", history=first.messages)

    assert first.messages[1] == {
        "role": "assistant",
        "content": "I am thinking about it",
        "tool_calls": [],
        "reasoning": [{"id": "rs_0", "summary": ["The user greets."], "encrypted_content": "encrypted-0"}],
        "response_id": "resp_1",
        "model": "test-model",
    }
    assert fake_openai.requests[1].body["input"] == [
        {"role": "user", "content": "Hi"},
        {
            "type": "reasoning",
            "id": "rs_0",
            "summary": [{"type": "summary_text", "text": "The user greets."}],
            "encrypted_content": "encrypted-0",
        },
        {"role": "assistant", "content": "I am thinking about it"},
        {"role": "user", "content": "And now?"},
    ]


async def test_reasoning_is_only_sent_back_to_the_model_that_wrote_it(
    fake_openai: FakeOpenAI, make_responses_agent: MakeAgent
) -> None:
    fake_openai.queue(responses_answer("Hello", reasoning=("Greet back.",)), responses_answer("Fine"))

    first = await make_responses_agent(model="model-a").run("Hi")
    await make_responses_agent(model="model-b").run("And now?", history=first.messages)

    sent_types = [item.get("type", "message") for item in fake_openai.requests[1].body["input"]]
    assert sent_types == ["message", "message", "message"]


async def test_encrypted_reasoning_can_be_turned_off(fake_openai: FakeOpenAI, make_responses_agent: MakeAgent) -> None:
    fake_openai.queue(responses_answer("Hi"))

    # Models that do not reason (e.g. gpt-4.1) reject the `include`.
    await make_responses_agent(encrypted_reasoning=False).run("Hi")

    assert "include" not in fake_openai.requests[0].body


# Stored responses (`store=True`) ------------------------------------------------------------------


async def test_stored_runs_send_only_new_messages(fake_openai: FakeOpenAI, make_responses_agent: MakeAgent) -> None:
    fake_openai.queue(
        responses_answer(tool_calls=(tool_call("add", a=2, b=3),), response_id="resp_1"),
        responses_answer("It is 5", response_id="resp_2"),
        responses_answer("You asked about 2 + 3", response_id="resp_3"),
    )
    agent = make_responses_agent(tools=[add], store=True)

    first = await agent.run("2 + 3?")
    await agent.run("What did I ask?", history=first.messages)

    first_request, tool_result_request, next_run_request = (request.body for request in fake_openai.requests)
    assert first_request["store"] is True
    assert "previous_response_id" not in first_request
    # Inside the run: only the tool result, on top of the stored response.
    assert tool_result_request["previous_response_id"] == "resp_1"
    assert tool_result_request["input"] == [{"type": "function_call_output", "call_id": "call_0", "output": "5"}]
    # The next run continues from the newest response in its history.
    assert next_run_request["previous_response_id"] == "resp_2"
    assert next_run_request["input"] == [{"role": "user", "content": "What did I ask?"}]


async def test_a_run_can_continue_from_a_response_id_alone(
    fake_openai: FakeOpenAI, make_responses_agent: MakeAgent
) -> None:
    fake_openai.queue(responses_answer("Hi Dani", response_id="resp_1"), responses_answer("Dani"))
    agent = make_responses_agent(store=True)

    first = await agent.run("I am Dani")
    second = await agent.run("My name?", previous_response_id=first.response_id)

    assert second.text == "Dani"
    request = fake_openai.requests[1].body
    assert (request["previous_response_id"], request["input"]) == ("resp_1", [{"role": "user", "content": "My name?"}])


async def test_a_lost_stored_response_is_replaced_by_the_whole_history(
    fake_openai: FakeOpenAI, make_responses_agent: MakeAgent
) -> None:
    history: list[Message] = [
        {"role": "user", "content": "I am Dani"},
        {"role": "assistant", "content": "Hi Dani", "response_id": "resp_old"},
    ]
    not_found = '{"error": {"code": "previous_response_not_found", "message": "Previous response not found."}}'
    fake_openai.queue(ErrorResponse(400, not_found), responses_answer("Dani"))

    result = await make_responses_agent(store=True).run("My name?", history=history)

    assert result.text == "Dani"
    retry = fake_openai.requests[1].body
    assert "previous_response_id" not in retry
    assert [item["content"] for item in retry["input"]] == ["I am Dani", "Hi Dani", "My name?"]


async def test_a_lost_response_given_by_id_raises(fake_openai: FakeOpenAI, make_responses_agent: MakeAgent) -> None:
    fake_openai.queue(error_response(400, "previous_response_not_found"))

    # There is no history to fall back to.
    with pytest.raises(ApiError, match="previous_response_not_found"):
        await make_responses_agent(store=True).run("Hi", previous_response_id="resp_gone")
    assert len(fake_openai.requests) == 1


async def test_messages_changed_by_hooks_are_all_sent(fake_openai: FakeOpenAI, make_responses_agent: MakeAgent) -> None:
    history: list[Message] = [
        {"role": "user", "content": "secret: 1234"},
        {"role": "assistant", "content": "Noted", "response_id": "resp_1"},
    ]

    def hide_secrets(ctx: RunContext, messages: list[Message]) -> list[Message]:
        return [
            {"role": "user", "content": "[hidden]"} if "secret" in str(message["content"]) else message
            for message in messages
        ]

    fake_openai.queue(responses_answer("Ok"))

    await make_responses_agent(store=True, hooks=[Hooks(before_model_request=hide_secrets)]).run("Hi", history=history)

    # OpenAI's stored copy still has the secret, so the request cannot build on it.
    request = fake_openai.requests[0].body
    assert "previous_response_id" not in request
    assert [item["content"] for item in request["input"]] == ["[hidden]", "Noted", "Hi"]


# Errors and invalid use ---------------------------------------------------------------------------


async def test_failed_responses_raise(fake_openai: FakeOpenAI, make_responses_agent: MakeAgent) -> None:
    failed = {"type": "response.failed", "response": {"id": "resp_1", "error": {"message": "server overloaded"}}}
    fake_openai.queue(StreamResponse([failed], end_with_done=False))

    with pytest.raises(ApiError, match="server overloaded"):
        await make_responses_agent().run("Hi")


async def test_a_stream_cut_before_the_response_completes_raises(
    fake_openai: FakeOpenAI, make_responses_agent: MakeAgent
) -> None:
    cut = StreamResponse([{"type": "response.output_text.delta", "delta": "Hel"}], end_with_done=False)
    fake_openai.queue(cut)

    with pytest.raises(InvalidResponseError, match="ended unexpectedly"):
        await make_responses_agent().run("Hi")


def test_invalid_configuration(make_agent: MakeAgent, make_responses_agent: MakeAgent) -> None:
    with pytest.raises(ValueError, match="Unknown api"):
        make_agent(api="completions")
    with pytest.raises(ValueError, match='needs `api="responses"`'):
        make_agent(store=True)

    stateless_agent, stored_agent = make_responses_agent(), make_responses_agent(store=True)
    with pytest.raises(ValueError, match="`store=True`"):
        stateless_agent.execute("Hi", previous_response_id="resp_1")
    with pytest.raises(ValueError, match="not both"):
        stored_agent.execute("Hi", previous_response_id="resp_1", history=[])
    with pytest.raises(ValueError, match="needs a prompt"):
        stored_agent.execute(previous_response_id="resp_1")


async def test_chat_history_continues_on_the_responses_api(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, make_responses_agent: MakeAgent
) -> None:
    fake_openai.queue(text_response("Hi Dani"), responses_answer("Dani"))

    first = await make_agent().run("I am Dani")
    await make_responses_agent().run("My name?", history=first.messages)

    assert [item["content"] for item in fake_openai.requests[1].body["input"]] == ["I am Dani", "Hi Dani", "My name?"]
