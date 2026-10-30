# pyox-agents

Minimal and fast AI agents for Python, with a Rust core. Inspired by [pi](https://github.com/earendil-works/pi).

You define the agent (model, prompt, tools) in Python and iterate over its events.
Rust does the heavy lifting: HTTP, streaming decoding, JSON parsing and the agent loop.

!!! note "Status: early"
    It supports the OpenAI Chat Completions API (and compatible servers, like Ollama), the
    [Responses API](responses-api.md), the [Decisions API](decisions.md), and Azure OpenAI.

## Install

pyox-agents is not on PyPI yet. Install it from GitHub; it compiles the Rust core, so you need a
Rust toolchain ([rustup](https://rustup.rs/)):

```bash
uv add "pyox-agents @ git+https://github.com/daniprol/pyox-agents"
```

## Your first agent

```python
import asyncio

from pyox_agents import Agent, TextDelta, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"Sunny in {city}"


agent = Agent(model="gpt-5-mini", prompt="Be concise.", tools=[get_weather])


async def main() -> None:
    async for event in agent.execute("Weather in Vigo?"):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)


asyncio.run(main())
```

```bash
export OPENAI_API_KEY=sk-...
uv run first_agent.py
```

The model calls `get_weather`, gets `"Sunny in Vigo"` back and streams its answer.

Don't need streaming? `await agent.run(...)` runs to the end and returns the final event:

```python
result = await agent.run("Weather in Vigo?")
print(result.text)
```

## Next

- [Streaming](streaming.md): the events of a run and how to group streamed text.
- [Tools](tools.md): Python functions, built-in Rust tools and your own Rust tools.
- [Images and files](images-and-files.md): send pictures and PDFs in a prompt.
- [History](history.md): continue conversations and store them in a database.
- [Hooks](hooks.md): watch or change a run at fixed points.
- [Client and errors](client.md): API keys, other servers, timeouts and retries.
- Complete, runnable scripts are in [`examples/`](https://github.com/daniprol/pyox-agents/tree/main/examples).
