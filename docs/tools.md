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
- Results that are not `str` are sent to the model as JSON.
- If a tool raises, the model gets the error message (and `ToolFinished.is_error` is `True`). The run continues.

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

## Built-in Rust tools

Compiled into the library and run entirely in Rust:

```python
from pyox_agents import Agent, BuiltinTool

agent = Agent(model="gpt-5-mini", tools=[BuiltinTool("read_file"), BuiltinTool("write_file")])
BuiltinTool.available()  # ['read_file', 'write_file']
```

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
