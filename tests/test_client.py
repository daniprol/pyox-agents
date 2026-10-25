import pytest

from conftest import MakeAgent
from factories import text_response
from fake_openai import FakeOpenAI
from pyox_agents import Agent, Client


async def test_reads_api_key_and_base_url_from_environment(
    fake_openai: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", fake_openai.base_url + "/")
    fake_openai.queue(text_response("ok"))

    await Agent(model="m").run("Hi")

    assert Client().base_url == fake_openai.base_url  # trailing slash removed
    assert fake_openai.requests[0].headers["authorization"] == "Bearer env-key"


def test_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert Client(api_key="x").base_url == "https://api.openai.com/v1"


async def test_one_client_shared_by_several_agents(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("one"), text_response("two"))

    first, second = make_agent(model="model-a"), make_agent(model="model-b")

    assert (await first.run("Hi")).text == "one"
    assert (await second.run("Hi")).text == "two"
    assert [request.body["model"] for request in fake_openai.requests] == ["model-a", "model-b"]
