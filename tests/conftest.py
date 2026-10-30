import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from fake_openai import FakeOpenAI
from pyox_agents import Agent, Client

MakeAgent = Callable[..., Agent]


@pytest.fixture
def fake_openai() -> Iterator[FakeOpenAI]:
    server = FakeOpenAI()
    yield server
    server.close()


@pytest.fixture
def client(fake_openai: FakeOpenAI) -> Client:
    return Client(api_key="test-key", base_url=fake_openai.base_url, headers={"X-Test": "yes"})


@pytest.fixture
def websocket_client(fake_openai: FakeOpenAI) -> Client:
    """Like `client`, sending Responses API requests over WebSocket."""
    return Client(api_key="test-key", base_url=fake_openai.base_url, headers={"X-Test": "yes"}, transport="websocket")


@pytest.fixture
def make_agent(client: Client) -> MakeAgent:
    """Factory for agents connected to the fake server: `make_agent(tools=[...])`."""

    def make(*, model: str = "test-model", **options: Any) -> Agent:
        return Agent(client=client, model=model, **options)

    return make


PROTOCOL_CLIENTS = Path(__file__).parent / "protocol_clients"

CheckStream = Callable[..., Any]


@pytest.fixture
def protocol_client() -> CheckStream:
    """Checks a stream with the official JavaScript client: `protocol_client("ai-sdk", sse_text)`.

    Returns what the client built from it (see `tests/protocol_clients/check.mjs`). Needs Node
    and `npm ci --prefix tests/protocol_clients`; skipped without them, except on CI.
    """
    if shutil.which("node") is None or not (PROTOCOL_CLIENTS / "node_modules").exists():
        if os.environ.get("CI"):
            pytest.fail("CI must check streams with the official clients: run `npm ci --prefix tests/protocol_clients`")
        pytest.skip("needs node and `npm ci --prefix tests/protocol_clients`")

    def check(protocol: str, sse: str, message: dict[str, Any] | None = None) -> Any:
        result = subprocess.run(
            ["node", "check.mjs", protocol],
            input=json.dumps({"sse": sse, "message": message}),
            cwd=PROTOCOL_CLIENTS,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    return check
