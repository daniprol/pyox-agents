"""Use tools written in Rust and compiled as a separate package (see rust_tool_package/).

uv pip install ./examples/rust_tool_package
OPENAI_API_KEY=... uv run --no-sync examples/rust_tools.py
"""

import asyncio

import fast_tools  # ty: ignore[unresolved-import]  (built separately, see the docstring)

from pyox_agents import Agent, Tool

TEXT_PARAMETER = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

count_words = Tool(
    name="count_words",
    description="Count the words in a text.",
    parameters=TEXT_PARAMETER,
    function=fast_tools.count_words,
    run_in_thread=True,  # heavy work runs in a worker thread, not on the event loop
)

most_common_words = Tool(
    name="most_common_words",
    description="The 5 most frequent words in a text, with their counts.",
    parameters=TEXT_PARAMETER,
    function=fast_tools.most_common_words,
    run_in_thread=True,
)

agent = Agent(model="gpt-5-mini", tools=[count_words, most_common_words])

result = asyncio.run(agent.run("How many words, and which are most common, in: 'to be or not to be'?"))
print(result.text)
