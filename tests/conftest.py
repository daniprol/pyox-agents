from collections.abc import Callable, Iterator
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
def make_agent(client: Client) -> MakeAgent:
    """Factory for agents connected to the fake server: `make_agent(tools=[...])`."""

    def make(*, model: str = "test-model", **options: Any) -> Agent:
        return Agent(client=client, model=model, **options)

    return make
