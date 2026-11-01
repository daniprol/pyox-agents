# pyox-agents

The fastest, lightest way to build AI agents in Python: a small Python API on top of a Rust core. Inspired by
[pi](https://github.com/earendil-works/pi).

You define the agent (model, instructions, tools) in Python and iterate over its events.
Rust does the heavy lifting: HTTP, streaming decoding, JSON parsing and the agent loop.

!!! note "Status: early"
    It supports the OpenAI Chat Completions API (and compatible servers, like Ollama), the
    [Responses API](responses-api.md), the [Decisions API](decisions.md), and Azure OpenAI.

## Install

```bash
uv add pyox-agents  # or: pip install pyox-agents
```

Prebuilt wheels cover Linux, macOS and Windows on Python 3.11 and newer, and pyox-agents has no Python
dependencies.

## Your first agent

```python
import asyncio

from pyox_agents import Agent, TextDelta, tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"Sunny in {city}"


agent = Agent(model="gpt-5-mini", instructions="Be concise.", tools=[get_weather])


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

- [Streaming](streaming.md): the events of a run, how it ends (and `run.stop()`), and grouping streamed text.
- [Tools](tools.md): Python functions and your own Rust tools.
- [Structured output](structured-output.md): answers as dataclasses, TypedDicts or pydantic models.
- [Images and files](images-and-files.md): send pictures and PDFs in a prompt, and get them from tools.
- [History](history.md): continue conversations and store them in a database.
- [Hooks](hooks.md): watch or change a run at fixed points.
- [Tracing](telemetry.md): OpenTelemetry spans and metrics for runs, model requests and tool calls.
- [Client and errors](client.md): API keys, other servers, timeouts and retries.
- Using a coding agent? Install the [pyox-agents skill](https://github.com/daniprol/pyox-agents/tree/main/skills/pyox-agents)
  with `npx skills add daniprol/pyox-agents`.
- Complete, runnable scripts are in [`examples/`](https://github.com/daniprol/pyox-agents/tree/main/examples).
