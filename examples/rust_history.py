"""Keep conversations in SQLite with a history store written in Rust (see rust_history_store/)."""

import asyncio

import fast_history

from pyox_agents import Agent

# Once, at startup: the store owns the database connection pool.
store = fast_history.sqlite_history("sqlite://chats.db")
agent = Agent(model="gpt-5-mini", prompt="Be brief.", history_store=store)


async def main() -> None:
    # Run this script twice: the second time, the agent still remembers.
    for prompt in ["My favourite city is Vigo.", "What is my favourite city?"]:
        result = await agent.run(prompt, conversation_id="demo")
        print(f"> {prompt}\n{result.text}\n")
    print(f"The conversation has {len(store.load('demo'))} messages.")


asyncio.run(main())
