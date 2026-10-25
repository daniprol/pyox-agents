"""A history store written in Rust (examples/rust_history_store, built by `uv sync`)."""

import asyncio
import sqlite3
from pathlib import Path

import fast_history
import pytest

from conftest import MakeAgent
from factories import text_response
from fake_openai import FakeOpenAI
from pyox_agents import HistoryError


@pytest.fixture
def database_url(tmp_path: Path) -> str:
    return f"sqlite://{tmp_path / 'chats.db'}"


async def test_conversation_survives_a_new_store(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, database_url: str
) -> None:
    fake_openai.queue(text_response("Hi Dani"), text_response("Your name is Dani"))

    await make_agent(history_store=fast_history.sqlite_history(database_url)).run("I am Dani", conversation_id="c1")
    # A new store (as after a restart) reads what the first one saved.
    store = fast_history.sqlite_history(database_url)
    result = await make_agent(history_store=store).run("My name?", conversation_id="c1")

    assert [message["content"] for message in fake_openai.requests[1].body["messages"]] == [
        "I am Dani",
        "Hi Dani",
        "My name?",
    ]
    assert store.load("c1") == result.messages
    assert store.load("unknown") == []


async def test_concurrent_conversations_share_one_pool(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, database_url: str
) -> None:
    fake_openai.queue(*(text_response("answer") for _ in range(8)))
    agent = make_agent(history_store=fast_history.sqlite_history(database_url, max_connections=2))

    await asyncio.gather(*(agent.run(f"question {number}", conversation_id=f"c{number}") for number in range(8)))

    store = fast_history.sqlite_history(database_url)
    assert [store.load(f"c{number}")[0]["content"] for number in range(8)] == [f"question {n}" for n in range(8)]


async def test_store_errors_raise_history_error(make_agent: MakeAgent, tmp_path: Path) -> None:
    database_file = tmp_path / "chats.db"
    store = fast_history.sqlite_history(f"sqlite://{database_file}")
    with sqlite3.connect(database_file) as connection:
        connection.execute("DROP TABLE messages")

    with pytest.raises(HistoryError, match="no such table"):
        await make_agent(history_store=store).run("Hi", conversation_id="c1")


def test_invalid_url_fails_early() -> None:
    with pytest.raises(RuntimeError):
        fast_history.sqlite_history("postgres://not-sqlite")
