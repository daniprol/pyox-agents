"""The Decisions API (`decide(...)`): requests, answers and errors."""

import pytest

from conftest import MakeAgent
from factories import decision_response, predicate_answer, tool_call, tool_calls_response
from fake_openai import ErrorResponse, FakeOpenAI, JsonResponse
from pyox_agents import (
    ApprovalRequired,
    Choice,
    ChoiceAnswer,
    Client,
    Hooks,
    InvalidResponseError,
    Message,
    Predicate,
    PredicateAnswer,
    RateLimitError,
    Refusal,
    RunContext,
    Score,
    ScoreAnswer,
    ToolCall,
    Usage,
    decide,
    image,
)


async def test_sends_the_questions_and_reads_every_kind_of_answer(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.queue(
        decision_response(
            predicate_answer("urgent", 0.97),
            {
                "type": "choice",
                "name": "team",
                "choice": "infra",
                "confidence": 0.9,
                "probabilities": [{"value": "infra", "probability": 0.9}, {"value": "billing", "probability": 0.1}],
            },
            {
                "type": "score",
                "name": "severity",
                "score": 1.8,
                "confidence": 0.8,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.0},
                    {"value": 1, "label": "Degraded", "probability": 0.2},
                    {"value": 2, "label": "Outage", "probability": 0.8},
                ],
            },
            {"type": "refusal", "name": "customer_name"},
            request_id="req_42",
        )
    )

    decision = await decide(
        "The deploy failed twice and customers are seeing 500s.",
        {
            "urgent": Predicate("Does this need attention right now?"),
            "team": Choice("Which team should handle it?", {"infra": "Deploys and outages.", "billing": "Payments."}),
            "severity": Score("How severe is the impact?", ["Cosmetic", "Degraded", "Outage"]),
            "customer_name": Predicate("Is the customer called Ana?"),
        },
        client=client,
    )

    request = fake_openai.requests[0]
    assert request.path == "/v1/decisions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.body == {
        "model": "gpt-6-luna",
        "input": "The deploy failed twice and customers are seeing 500s.",
        "questions": [
            {"type": "predicate", "name": "urgent", "instructions": "Does this need attention right now?"},
            {
                "type": "choice",
                "name": "team",
                "instructions": "Which team should handle it?",
                "choices": [
                    {"value": "infra", "description": "Deploys and outages."},
                    {"value": "billing", "description": "Payments."},
                ],
            },
            {
                "type": "score",
                "name": "severity",
                "instructions": "How severe is the impact?",
                "levels": [{"label": "Cosmetic"}, {"label": "Degraded"}, {"label": "Outage"}],
            },
            {"type": "predicate", "name": "customer_name", "instructions": "Is the customer called Ana?"},
        ],
    }

    urgent, team, severity, customer_name = (decision[name] for name in decision.answers)
    assert isinstance(urgent, PredicateAnswer)
    assert urgent.probability == 0.97
    assert isinstance(team, ChoiceAnswer)
    assert (team.choice, team.confidence, team.probabilities) == ("infra", 0.9, {"infra": 0.9, "billing": 0.1})
    assert isinstance(severity, ScoreAnswer)
    assert (severity.score, severity.confidence) == (1.8, 0.8)
    assert severity.probabilities == {"Cosmetic": 0.0, "Degraded": 0.2, "Outage": 0.8}
    assert isinstance(customer_name, Refusal)
    assert (decision.model, decision.request_id, len(decision)) == ("gpt-6-luna", "req_42", 4)
    assert decision.usage == Usage(input_tokens=42, requests=1)


async def test_choices_can_be_booleans_and_options_without_descriptions(
    fake_openai: FakeOpenAI, client: Client
) -> None:
    probabilities = [{"value": True, "probability": 0.3}, {"value": False, "probability": 0.7}]
    choice = {"type": "choice", "name": "refund", "choice": False, "confidence": 0.7, "probabilities": probabilities}
    fake_openai.queue(decision_response(choice))

    decision = await decide("A refund request", {"refund": Choice("Refund it?", [True, False])}, client=client)

    assert fake_openai.requests[0].body["questions"][0]["choices"] == [{"value": True}, {"value": False}]
    answer = decision["refund"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice is False
    assert answer.probabilities == {True: 0.3, False: 0.7}


async def test_optional_request_fields_are_sent_only_when_given(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.queue(decision_response(predicate_answer("ok", 0.5)))

    await decide(
        "Hi",
        {"ok": Predicate("Is it ok?")},
        client=client,
        model="gpt-6-luna-mini",
        safety_identifier="user-hash",
        prompt_cache_key="triage",
    )

    body = fake_openai.requests[0].body
    assert (body["model"], body["safety_identifier"], body["prompt_cache_key"]) == (
        "gpt-6-luna-mini",
        "user-hash",
        "triage",
    )


# Inputs -------------------------------------------------------------------------------------------


async def test_a_prompt_with_images_becomes_one_user_message(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.queue(decision_response(predicate_answer("damaged", 0.9)))

    await decide(
        ["Is the screen broken?", image("https://example.com/phone.png")],
        {"damaged": Predicate("Is the item damaged?")},
        client=client,
    )

    assert fake_openai.requests[0].body["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Is the screen broken?"},
                {"type": "input_image", "image_url": "https://example.com/phone.png"},
            ],
        }
    ]


async def test_a_history_becomes_input_items_without_reasoning(fake_openai: FakeOpenAI, client: Client) -> None:
    history: list[Message] = [
        {"role": "user", "content": "Delete my old files"},
        {
            "role": "assistant",
            "content": "",
            "model": "gpt-6-luna",
            "reasoning": [{"id": "rs_1", "summary": [], "encrypted_content": "secret"}],
            "tool_calls": [{"id": "call_1", "name": "delete", "arguments": {"path": "/old"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "delete", "content": "deleted"},
        {"role": "assistant", "content": "Done."},
    ]
    fake_openai.queue(decision_response(predicate_answer("asked", 0.99)))

    await decide(history, {"asked": Predicate("Did the user ask for the deletion?")}, client=client)

    assert fake_openai.requests[0].body["input"] == [
        {"role": "user", "content": "Delete my old files"},
        {"type": "function_call", "call_id": "call_1", "name": "delete", "arguments": '{"path":"/old"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "deleted"},
        {"role": "assistant", "content": "Done."},
    ]


# Errors -------------------------------------------------------------------------------------------


async def test_api_errors_raise_the_usual_exceptions(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.queue(ErrorResponse(429, '{"error": {"message": "Slow down"}}'))

    with pytest.raises(RateLimitError, match="Slow down"):
        await decide("Hi", {"ok": Predicate("Is it ok?")}, client=client)


async def test_an_answer_of_another_shape_raises(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.queue(JsonResponse({"answers": "none"}))

    with pytest.raises(InvalidResponseError):
        await decide("Hi", {"ok": Predicate("Is it ok?")}, client=client)


async def test_an_unknown_question_name_raises_key_error(fake_openai: FakeOpenAI, client: Client) -> None:
    fake_openai.queue(decision_response(predicate_answer("ok", 0.5)))

    decision = await decide("Hi", {"ok": Predicate("Is it ok?")}, client=client)

    with pytest.raises(KeyError, match="other"):
        decision["other"]


# With an agent ------------------------------------------------------------------------------------


async def test_a_hook_can_ask_for_approval_when_a_call_looks_risky(
    fake_openai: FakeOpenAI, client: Client, make_agent: MakeAgent
) -> None:
    # The recipe from docs/decisions.md.
    async def risk_check(ctx: RunContext, call: ToolCall) -> None:
        decision = await decide(
            f"Tool call: {call['name']}({call['arguments']})",
            {"risky": Predicate("Could this tool call cause harm or lose data?")},
            client=client,
        )
        answer = decision["risky"]
        if not isinstance(answer, PredicateAnswer) or answer.probability >= 0.5:
            raise ApprovalRequired({"tool": call["name"]})

    def delete(path: str) -> str:
        return f"deleted {path}"

    fake_openai.queue(
        tool_calls_response(tool_call("delete", path="/")),
        decision_response(predicate_answer("risky", 0.98)),
    )

    result = await make_agent(tools=[delete], hooks=[Hooks(before_tool_call=risk_check)]).run("Free some space")

    [request] = result.pending
    assert (request.kind, request.data) == ("approval", {"tool": "delete"})
    assert fake_openai.requests[1].body["input"] == "Tool call: delete({'path': '/'})"
