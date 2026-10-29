"""Use tools and hooks written in Rust and compiled as a separate package (see rust_tool_package/)."""

import asyncio

import fast_tools

from pyox_agents import Agent

# The tools describe their own arguments (generated from their Rust structs).
# The `guard` hooks block huge texts and send only the recent messages to the model.
agent = Agent(
    model="gpt-5-mini",
    tools=[fast_tools.count_words, fast_tools.most_common_words],
    hooks=[fast_tools.guard],
)

result = asyncio.run(agent.run("How many words, and which are most common, in: 'to be or not to be'?"))
print(result.text)
