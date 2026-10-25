from collections.abc import Iterator

import pytest
from fake_openai import FakeOpenAI

from pyox_agents import Client


@pytest.fixture
def fake_openai() -> Iterator[FakeOpenAI]:
    server = FakeOpenAI()
    yield server
    server.close()


@pytest.fixture
def client(fake_openai: FakeOpenAI) -> Client:
    return Client(api_key="test-key", base_url=fake_openai.base_url, headers={"X-Test": "yes"})
