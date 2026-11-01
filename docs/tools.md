# Tools

Tools are functions the model can call. A run calls them, sends the results back to the model and continues
until the model answers.

## Python tools

Decorate a function (sync or async) with `@tool`. The JSON schema comes from the type hints and the description
from the docstring:

```python
from typing import Annotated, Literal

from pyox_agents import Agent, tool


@tool
async def search(query: Annotated[str, "What to look for"], order: Literal["new", "old"] = "new") -> list[str]:
    """Search the documents."""
    return [f"Result for {query}"]


agent = Agent(model="gpt-5-mini", tools=[search])
```

- Plain functions passed in `tools=[...]` are wrapped with `@tool` automatically.
- Results that are not `str` are sent to the model as JSON. A tool can also return
  [images and files](images-and-files.md#tool-results).
- If a tool raises, the model gets the error message (and `ToolFinished.is_error` is `True`). The run continues.
- Sync tools never block your event loop: they run on worker threads (in [sync runs](sync.md), on your own thread,
  unless `@tool(run_in_thread=True)`).

Parameters can be `str`, `int`, `float`, `bool`, lists, dicts, `Literal`s, unions (`X | None`), dataclasses,
`TypedDict`s and pydantic models (anything with `model_json_schema()` and `model_validate()`; pydantic is not a
dependency). Dataclass and pydantic arguments arrive built from the model's JSON:

```python
from dataclasses import dataclass


@dataclass
class Address:
    city: str
    street: Annotated[str, "Street and number"] = ""


@tool
def ship(to: Address, express: bool = False) -> str:
    """Ship the parcel."""
    return f"Shipping to {to.city}"  # `to` is an `Address`
```

## Run context and dependencies

A tool whose first parameter is typed `RunContext` gets the run's context; the model never sees that parameter.
Pass what your tools need (a database pool, the current user...) as `deps`:

```python
from pyox_agents import RunContext


@tool
async def my_orders(ctx: RunContext[Database], status: str = "open") -> list[str]:
    """List the user's orders."""
    return await ctx.deps.orders(ctx.conversation_id, status)


await agent.run("What did I order?", deps=database)
```

`ctx` has `deps`, `conversation_id`, `turn`, `usage` (tokens used so far), and, inside a
tool, `tool_call_id`, `tool_name`, `approval` and the `ask` and `step` methods
(see [Human in the loop](human-in-the-loop.md)). [Hooks](hooks.md) get the same `ctx`.

## Tools for one run

`tools=` on `run(...)` or `execute(...)` adds tools for that run only, on top of the agent's. Use it when the
tools themselves depend on the request: a user's own [MCP server](mcp.md), or tools made for their plan.

```python
agent = Agent(model="gpt-5-mini", tools=[search])  # made once, for every user

result = await agent.run(prompt, tools=tools_for(user))
```

Their names must differ from the agent's tools. A run that stopped to wait for a person needs the same tools
again when it goes on. Subagents made with `inherit_tools=True` get them too. To hide some of the agent's tools
instead, use a [`before_model_request` hook](hooks.md).

## Your own Rust tools

For tools that must be fast, write them in Rust in your own small package. Describe the arguments with a struct
and implement the `RustTool` trait from the
[`pyox-tools`](https://github.com/daniprol/pyox-agents/tree/main/crates/pyox-tools) crate. The JSON schema is
generated from the struct (doc comments become descriptions):

```rust
/// Count the words in a text.
#[derive(Deserialize, JsonSchema)]
struct CountWords {
    /// The text to count words in.
    text: String,
}

impl RustTool for CountWords {
    const NAME: &'static str = "count_words";

    fn run(self) -> Result<String, String> {
        Ok(self.text.split_whitespace().count().to_string())
    }
}
```

Register it in the module with `add_tool::<CountWords>(module)?`, then use it from Python:

```python
--8<-- "examples/rust_tools.py"
```

When the model calls these tools, the agent runs your Rust function directly: no Python is involved.
Return `Err(message)` to tell the model something went wrong.

[`examples/rust_tool_package`](https://github.com/daniprol/pyox-agents/tree/main/examples/rust_tool_package)
is a complete package to copy. Try it from the repository root:

```bash
uv sync
uv run examples/rust_tools.py
```

## Any object

Any object with `name`, `description`, `parameters` (a JSON schema) and an `invoke(arguments)` method (sync or
async) works as a tool too.
