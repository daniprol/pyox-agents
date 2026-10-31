"""`Agent(output_type=...)`: answers as dataclasses, TypedDicts, pydantic models or schema dicts."""

import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

import pytest

from conftest import MakeAgent
from factories import responses_answer, text_response
from fake_openai import FakeOpenAI, StreamResponse
from pyox_agents import InvalidResponseError
from pyox_agents.schema import output_format


@dataclass
class Step:
    action: str


@dataclass
class Ticket:
    team: Literal["billing", "support"]
    summary: Annotated[str, "One sentence"]
    assignee: str | None = None
    steps: list[Step] = field(default_factory=list)


TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "team": {"type": "string", "enum": ["billing", "support"]},
        "summary": {"type": "string", "description": "One sentence"},
        "assignee": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"action": {"type": "string"}},
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    },
    # Strict mode: every field is required; `assignee` may be null.
    "required": ["team", "summary", "assignee", "steps"],
    "additionalProperties": False,
}

TICKET_JSON = json.dumps(
    {"team": "support", "summary": "Printer on fire", "assignee": None, "steps": [{"action": "Call 112"}]}
)


async def test_chat_asks_for_a_strict_schema_and_parses_a_dataclass(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    fake_openai.queue(text_response(TICKET_JSON))

    result = await make_agent(output_type=Ticket).run("The printer is on fire")

    assert fake_openai.requests[0].body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Ticket", "schema": TICKET_SCHEMA, "strict": True},
    }
    assert result.output == Ticket(team="support", summary="Printer on fire", steps=[Step("Call 112")])
    assert result.text == TICKET_JSON


async def test_responses_api_asks_with_text_format(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(responses_answer(TICKET_JSON))

    agent = make_agent(api="responses", output_type=Ticket, model_parameters={"text": {"verbosity": "low"}})
    result = await agent.run("The printer is on fire")

    assert fake_openai.requests[0].body["text"] == {
        "verbosity": "low",
        "format": {"type": "json_schema", "name": "Ticket", "schema": TICKET_SCHEMA, "strict": True},
    }
    assert isinstance(result.output, Ticket)


class Weather(TypedDict):
    city: str
    celsius: float


@pytest.mark.parametrize(
    "output_type",
    [
        Weather,
        {"title": "weather report", "type": "object", "properties": {"city": {"type": "string"}}},
    ],
)
async def test_typeddicts_and_schema_dicts_give_dicts(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, output_type: Any
) -> None:
    fake_openai.queue(text_response('{"city": "Vigo", "celsius": 18.5}'))

    result = await make_agent(output_type=output_type).run("Weather?")

    assert result.output == {"city": "Vigo", "celsius": 18.5}
    name = fake_openai.requests[0].body["response_format"]["json_schema"]["name"]
    assert name in ("Weather", "weather_report")  # names allow letters, digits, `_` and `-` only


async def test_pydantic_models_by_duck_typing(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    pydantic = pytest.importorskip("pydantic")

    class City(pydantic.BaseModel):
        name: str
        country: str | None = None

    class Trip(pydantic.BaseModel):
        cities: list[City]

    fake_openai.queue(text_response('{"cities": [{"name": "Vigo", "country": null}]}'))

    result = await make_agent(output_type=Trip).run("Plan a trip")

    assert result.output == Trip(cities=[City(name="Vigo")])
    schema = fake_openai.requests[0].body["response_format"]["json_schema"]["schema"]
    assert schema["$defs"]["City"]["required"] == ["name", "country"]
    assert "default" not in schema["$defs"]["City"]["properties"]["country"]  # strict mode rejects it


async def test_an_answer_not_matching_raises_with_the_messages(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Not JSON at all"))

    with pytest.raises(InvalidResponseError, match="does not match `output_type`") as error:
        await make_agent(output_type=Ticket).run("Hi")

    assert error.value.messages is not None
    assert error.value.messages[-1]["content"] == "Not JSON at all"


async def test_no_output_when_the_answer_is_cut_or_refused(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        StreamResponse(
            [
                {"choices": [{"delta": {"content": '{"team": "sup'}}]},
                {"choices": [{"delta": {}, "finish_reason": "length"}]},
            ]
        ),
        StreamResponse([{"choices": [{"delta": {"refusal": "I can't help with that."}, "finish_reason": "stop"}]}]),
    )
    agent = make_agent(output_type=Ticket)

    cut, refused = await agent.run("Hi"), await agent.run("Hi")

    assert (cut.stop_reason, cut.output) == ("length", None)
    assert (refused.stop_reason, refused.output) == ("refusal", None)


async def test_without_output_type_there_is_no_output(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hi"))

    result = await make_agent().run("Hi")

    assert result.output is None
    assert "response_format" not in fake_openai.requests[0].body


def test_output_type_must_be_an_object() -> None:
    with pytest.raises(TypeError, match="JSON object"):
        output_format(list[str])
