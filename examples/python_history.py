"""Keep conversations in SQLite with a history store written in Python, using the SQLAlchemy ORM.

Compare with rust_history.py: same idea, but the store is Python code.
"""

import asyncio
import json
from typing import Any

from sqlalchemy import JSON, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from pyox_agents import Agent


class Base(DeclarativeBase):
    pass


class StoredMessage(Base):
    """One row per message, so saving a run only appends what it added."""

    __tablename__ = "messages"
    conversation_id: Mapped[str] = mapped_column(primary_key=True)
    position: Mapped[int] = mapped_column(primary_key=True)
    message: Mapped[dict[str, Any]] = mapped_column(JSON)


class SqlAlchemyHistory:
    """A history store: the agent calls `load` before each run and `save` after it."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.sessions = async_sessionmaker(engine)  # uses the engine's connection pool

    async def load(self, conversation_id: str) -> list[Any]:
        async with self.sessions() as session:
            messages = await session.scalars(
                select(StoredMessage.message)
                .where(StoredMessage.conversation_id == conversation_id)
                .order_by(StoredMessage.position)
            )
            return list(messages)

    async def save(self, conversation_id: str, new_messages_json: str) -> None:
        async with self.sessions.begin() as session:  # one transaction
            saved_count = await session.scalar(
                select(func.count()).where(StoredMessage.conversation_id == conversation_id)
            )
            session.add_all(
                StoredMessage(conversation_id=conversation_id, position=(saved_count or 0) + offset, message=message)
                for offset, message in enumerate(json.loads(new_messages_json))
            )


async def main() -> None:
    # Once, at startup: the engine owns the database connection pool.
    engine = create_async_engine("sqlite+aiosqlite:///chats_sqlalchemy.db")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    store = SqlAlchemyHistory(engine)
    agent = Agent(model="gpt-5-mini", prompt="Be brief.", history_store=store)

    # Run this script twice: the second time, the agent still remembers.
    for prompt in ["My favourite city is Vigo.", "What is my favourite city?"]:
        result = await agent.run(prompt, conversation_id="demo")
        print(f"> {prompt}\n{result.text}\n")
    print(f"The conversation has {len(await store.load('demo'))} messages.")
    await engine.dispose()


asyncio.run(main())
