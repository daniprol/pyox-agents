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
from factories import collect_events, events_of_type
from pyox_agents import AuthenticationError, BuiltinTool, Client, RunFinished, TextDelta, ToolFinished, tool

pytestmark = pytest.mark.live

DEFAULT_MODEL = "gpt-6-luna"


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


@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_calls_a_python_tool(make_agent: MakeAgent, api: str) -> None:
    secret = secrets.token_hex(4)

    @tool
    def get_secret() -> str:
        """Return the secret code."""
        return secret

    events = await collect_events(make_agent(api=api, tools=[get_secret]).execute("What is the secret code?"))

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
    agent = make_agent(tools=[BuiltinTool("read_file")])

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
