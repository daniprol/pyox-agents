import pytest

from conftest import MakeAgent
from factories import responses_answer, text_response
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


# Azure OpenAI (Microsoft Foundry) -----------------------------------------------------------------


async def test_azure_client_talks_to_the_v1_api(fake_openai: FakeOpenAI) -> None:
    # A resource endpoint such as https://my-resource.openai.azure.com; here the fake server.
    endpoint = fake_openai.base_url.removesuffix("/v1")
    fake_openai.queue(responses_answer("Hi"))
    agent = Agent(client=Client.azure(endpoint, api_key="azure-key"), model="my-deployment", api="responses")

    await agent.run("Hi")

    request = fake_openai.requests[0]
    assert request.path == "/openai/v1/responses"  # no `?api-version=...`
    assert request.body["model"] == "my-deployment"
    assert request.headers["authorization"] == "Bearer azure-key"


def test_azure_client_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my-resource.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")

    assert Client.azure().base_url == "https://my-resource.openai.azure.com/openai/v1"


def test_azure_client_needs_an_endpoint_and_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")  # never used for Azure

    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        Client.azure(api_key="azure-key")
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        Client.azure("https://my-resource.openai.azure.com")
