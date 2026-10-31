"""End-to-end checks against a real model on Azure OpenAI (Microsoft Foundry).

Skipped by default. Run them with `uv run pytest -m live`, with these set:

- `AZURE_OPENAI_ENDPOINT`: e.g. https://my-resource.openai.azure.com
- `AZURE_OPENAI_API_KEY`: the resource key
- `LIVE_MODEL` (optional): the deployment name, `gpt-6-luna` by default

Answers vary, so the tests check the shape of a run (events, tool calls, usage), not its wording.
Prompts are tiny to spend few tokens. CI runs them on pushes to main and dev (.github/workflows/ci.yml).
"""

import os
import secrets
from pathlib import Path

import pytest

from conftest import MakeAgent
from factories import collect_events, events_of_type, run_finished
from pyox_agents import (
    Answer,
    ApiError,
    Approve,
    AuthenticationError,
    BuiltinTool,
    Choice,
    ChoiceAnswer,
    Client,
    InputRequested,
    Predicate,
    PredicateAnswer,
    RunContext,
    RunFinished,
    StateChanged,
    SubagentFinished,
    SubagentStarted,
    TextDelta,
    ToolFinished,
    decide,
    tool,
)

pytestmark = pytest.mark.live

DEFAULT_MODEL = "gpt-6-luna"

# Reasoning models like gpt-6-luna accept function tools on Chat Completions only without reasoning.
NO_REASONING = {"reasoning_effort": "none"}


@pytest.fixture
def client() -> Client:
    """Overrides the fake server's client: reads AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."""
    return Client.azure(request_timeout_seconds=120)


@pytest.fixture
def model() -> str:
    return os.environ.get("LIVE_MODEL") or DEFAULT_MODEL


def assert_used_tokens(finished: RunFinished) -> None:
    assert finished.usage.input_tokens > 0
    assert finished.usage.output_tokens > 0


@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_answers_with_text(make_agent: MakeAgent, api: str) -> None:
    finished = await make_agent(api=api).run("Reply with the word: ready")

    assert finished.text.strip()
    assert finished.turns == 1
    assert_used_tokens(finished)


@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_streams_text(make_agent: MakeAgent, api: str) -> None:
    events = await collect_events(make_agent(api=api).execute("Count from 1 to 5, separated by spaces."))

    deltas = events_of_type(events, TextDelta)
    [finished] = events_of_type(events, RunFinished)
    assert deltas
    assert "".join(delta.text for delta in deltas) == finished.text


@pytest.mark.parametrize(
    ("api", "model_parameters"), [("chat", NO_REASONING), ("responses", None)], ids=["chat", "responses"]
)
async def test_calls_a_python_tool(make_agent: MakeAgent, api: str, model_parameters: dict[str, str] | None) -> None:
    secret = secrets.token_hex(4)

    @tool
    def get_secret() -> str:
        """Return the secret code."""
        return secret

    events = await collect_events(
        make_agent(api=api, model_parameters=model_parameters, tools=[get_secret]).execute("What is the secret code?")
    )

    [finished_tool] = events_of_type(events, ToolFinished)
    [finished] = events_of_type(events, RunFinished)
    assert finished_tool.name == "get_secret"
    assert not finished_tool.is_error
    assert finished.turns >= 2
    assert secret in finished.text


async def test_calls_a_rust_tool(make_agent: MakeAgent, tmp_path: Path) -> None:
    secret = secrets.token_hex(4)
    path = tmp_path / "secret.txt"
    path.write_text(secret)
    agent = make_agent(api="responses", tools=[BuiltinTool("read_file")])

    events = await collect_events(agent.execute(f"Read the file {path} and tell me what it says."))

    [finished_tool] = events_of_type(events, ToolFinished)
    [finished] = events_of_type(events, RunFinished)
    assert finished_tool.name == "read_file"
    assert finished_tool.result == secret
    assert secret in finished.text


async def test_continues_over_websocket(make_agent: MakeAgent) -> None:
    agent = make_agent(client=Client.azure(transport="websocket"), api="responses")

    first = await agent.run("Remember the number 42. Reply with: ok")
    second = await agent.run("Which number did I ask you to remember?", history=first.messages)

    assert "42" in second.text
    assert_used_tokens(second)


async def test_rejects_a_wrong_key(make_agent: MakeAgent) -> None:
    bad_client = Client.azure(api_key="not-a-real-key")

    with pytest.raises(AuthenticationError):
        await make_agent(client=bad_client).run("Hi")


async def test_answers_a_decision(client: Client, model: str) -> None:
    try:
        decision = await decide(
            "Customer: my order arrived with a broken screen.",
            {
                "damaged": Predicate("Does the customer report a damaged item?"),
                "mood": Choice("How does the customer feel?", ["happy", "upset"]),
            },
            client=client,
            model=model,
        )
    except ApiError as error:
        if "HTTP 404" in str(error):
            pytest.skip("This endpoint does not serve the Decisions API")
        raise

    damaged, mood = decision["damaged"], decision["mood"]
    assert isinstance(damaged, PredicateAnswer)
    assert damaged.probability > 0.5
    assert isinstance(mood, ChoiceAnswer)
    assert mood.choice == "upset"
    assert decision.usage.input_tokens > 0


# Subagents and state -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("api", "model_parameters"), [("chat", NO_REASONING), ("responses", None)], ids=["chat", "responses"]
)
async def test_delegates_to_a_subagent(
    make_agent: MakeAgent, api: str, model_parameters: dict[str, str] | None
) -> None:
    secret = secrets.token_hex(4)

    @tool
    def get_secret() -> str:
        """Return the secret code."""
        return secret

    options = {"api": api, "model_parameters": model_parameters}
    researcher = make_agent(**options, prompt="Use your tools to answer.", tools=[get_secret])
    agent = make_agent(
        **options,
        prompt="You cannot see the secret code. Always ask the `research` tool for it.",
        tools=[researcher.as_tool("research", "Finds the secret code.")],
    )

    events = await collect_events(agent.execute("What is the secret code?"))

    [started] = events_of_type(events, SubagentStarted)
    [finished_subagent] = events_of_type(events, SubagentFinished)
    assert started.name == "research"
    assert not finished_subagent.is_error
    assert finished_subagent.usage.requests >= 2
    subagent_tools = [event for event in events_of_type(events, ToolFinished) if event.subagent_id == started.call_id]
    assert [(event.name, event.result) for event in subagent_tools] == [("get_secret", secret)]
    finished = run_finished(events)
    assert secret in finished.text
    assert finished.usage.requests >= 4  # the subagent's requests count too


async def test_a_subagent_approval_stops_and_resumes_the_run(make_agent: MakeAgent) -> None:
    booked: list[str] = []

    @tool(requires_approval=True)
    def book(city: str) -> str:
        """Book a trip to a city."""
        booked.append(city)
        return f"Booked {city}"

    travel_agent = make_agent(api="responses", prompt="Book what you are asked with `book`.", tools=[book])
    agent = make_agent(
        api="responses",
        prompt="Use the `travel` tool to book trips.",
        tools=[travel_agent.as_tool("travel", "Books trips.")],
    )

    stopped = await agent.run("Book a trip to Vigo.")

    [request] = stopped.pending
    assert (request.kind, request.tool_name) == ("approval", "book")
    assert request.subagent_id
    assert request.call_id.startswith(f"{request.subagent_id}/")
    assert booked == []

    result = await agent.run(history=stopped.messages, answers={request.call_id: Approve()})

    assert result.pending == []
    assert [city.lower() for city in booked] == ["vigo"]
    assert result.text.strip()


async def test_a_subagent_asks_the_user(make_agent: MakeAgent) -> None:
    asked: list[InputRequested] = []

    def on_request(request: InputRequested) -> Answer:
        asked.append(request)
        return Answer("Dani")

    greeter = make_agent(
        api="responses",
        prompt="First call `ask_user` to ask for the user's name. Then reply with a greeting that uses it.",
    )
    agent = make_agent(
        api="responses",
        prompt="Use the `greet` tool, then repeat its greeting.",
        tools=[greeter.as_tool("greet", "Greets the user by name.", talk_to="user")],
    )

    result = await agent.run("Greet me.", on_request=on_request)

    [request] = asked
    assert (request.kind, request.tool_name) == ("input", "ask_user")
    assert request.subagent_id
    assert "Dani" in result.text


async def test_a_subagent_asks_the_calling_agent(make_agent: MakeAgent) -> None:
    planner = make_agent(
        api="responses",
        prompt=(
            "First call `ask_parent` to ask which city the trip is to. "
            "Then reply with only the city name, in capital letters."
        ),
    )
    agent = make_agent(
        api="responses",
        prompt="The trip is to Vigo. Use the `plan` tool, answer any question it asks, then repeat its final answer.",
        tools=[planner.as_tool("plan", "Plans a trip.", talk_to="parent")],
    )

    events = await collect_events(agent.execute("Plan my trip."))

    asks = [event for event in events_of_type(events, ToolFinished) if event.name == "ask_parent"]
    assert len(asks) == 1
    assert len(events_of_type(events, SubagentStarted)) >= 2  # it asked, then went on with the answer
    assert "VIGO" in run_finished(events).text.upper()


async def test_tools_change_the_state(make_agent: MakeAgent) -> None:
    @tool
    def add_todo(ctx: RunContext, text: str) -> str:
        """Add an item to the user's todo list."""
        ctx.state["todos"].append(text)
        return "Added"

    helper = make_agent(api="responses", prompt="Manage the todo list with your tools.", tools=[add_todo])
    agent = make_agent(
        api="responses",
        prompt="Use the `todos` tool to change the user's todo list.",
        tools=[helper.as_tool("todos", "Changes the todo list.", state="write")],
    )

    events = await collect_events(agent.execute("Add 'pack' to my todo list.", state={"todos": []}))

    [change] = events_of_type(events, StateChanged)
    assert change.subagent_id
    [todo] = change.state["todos"]
    assert "pack" in todo.lower()
    assert run_finished(events).state == change.state
