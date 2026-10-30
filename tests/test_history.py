"""History: passing it directly, and loading/saving it with Python history stores."""

import json

import pytest

from conftest import MakeAgent
from factories import MemoryHistory, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import HistoryError, Message, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def sent_messages(fake_openai: FakeOpenAI, request_index: int = 0) -> list[dict]:
    return fake_openai.requests[request_index].body["messages"]


async def test_new_messages_and_json_cover_only_this_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hi Dani"), text_response("Dani"))
    agent = make_agent()
    first = await agent.run("I am Dani")

    second = await agent.run("My name?", history=first.messages)

    assert second.new_messages == [
        {"role": "user", "content": "My name?"},
        {"role": "assistant", "content": "Dani", "tool_calls": [], "model": "test-model"},
    ]
    assert json.loads(second.new_messages_json()) == second.new_messages
    assert json.loads(second.messages_json()) == second.messages
    assert len(second.messages) == 4


@pytest.mark.parametrize("encode", [str, str.encode], ids=["str", "bytes"])
async def test_history_can_be_json_text(fake_openai: FakeOpenAI, make_agent: MakeAgent, encode) -> None:
    fake_openai.queue(text_response("ok"))
    history = json.dumps([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])

    await make_agent().run("Bye", history=encode(history))

    assert [message["content"] for message in sent_messages(fake_openai)] == ["Hi", "Hello", "Bye"]


async def test_invalid_json_history_is_rejected(make_agent: MakeAgent) -> None:
    with pytest.raises(ValueError, match="Invalid history"):
        make_agent().execute("Hi", history="not json")


async def test_store_loads_and_saves_a_conversation(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("add", a=1, b=2)), text_response("3"), text_response("Yes"))
    store = MemoryHistory()
    agent = make_agent(tools=[add], history_store=store)

    await agent.run("1 + 2?", conversation_id="chat-1")
    result = await agent.run("Sure?", conversation_id="chat-1")

    assert [message["role"] for message in store.conversations["chat-1"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]
    assert store.conversations["chat-1"] == result.messages
    assert sent_messages(fake_openai, 2)[-1] == {"role": "user", "content": "Sure?"}
    assert len(sent_messages(fake_openai, 2)) == 5


async def test_sync_load_function_without_save(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Fine"))
    loaded_ids = []

    def load(conversation_id: str) -> str:
        loaded_ids.append(conversation_id)
        return '[{"role": "user", "content": "How are you?"}]'

    result = await make_agent(history_store=load).run(conversation_id="chat-2")

    assert loaded_ids == ["chat-2"]
    assert result.new_messages == [{"role": "assistant", "content": "Fine", "tool_calls": [], "model": "test-model"}]


async def test_a_run_without_conversation_id_skips_the_store(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("ok"))
    store = MemoryHistory()

    await make_agent(history_store=store).run("Hi")

    assert store.conversations == {}


async def test_empty_conversation_without_prompt_fails(make_agent: MakeAgent) -> None:
    agent = make_agent(history_store=MemoryHistory())

    with pytest.raises(ValueError, match='Conversation "new" has no messages yet'):
        await agent.run(conversation_id="new")


async def test_store_errors_reach_the_caller(make_agent: MakeAgent) -> None:
    class BrokenStore:
        async def load(self, conversation_id: str) -> list[Message]:
            raise ConnectionError("database is down")

    def load_invalid(conversation_id: str) -> str:
        return "not json"

    with pytest.raises(ConnectionError, match="database is down"):
        await make_agent(history_store=BrokenStore()).run("Hi", conversation_id="x")
    with pytest.raises(HistoryError, match='Could not read the history of "x"'):
        await make_agent(history_store=load_invalid).run("Hi", conversation_id="x")


async def test_wrong_arguments_are_rejected(make_agent: MakeAgent) -> None:
    with pytest.raises(TypeError, match="is not a history store"):
        make_agent(history_store=42)
    with pytest.raises(ValueError, match="needs an agent with a `history_store`"):
        make_agent().execute("Hi", conversation_id="x")
    with pytest.raises(ValueError, match="not both"):
        make_agent(history_store=MemoryHistory()).execute("Hi", history=[], conversation_id="x")
