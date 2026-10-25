# fast-tools: agent tools written in Rust

A minimal Python package whose tools are written in Rust, compiled separately
from pyox-agents. Copy this folder to start your own.

## Use it

```bash
uv pip install ./examples/rust_tool_package    # compile and install (needs Rust)
```

```python
import fast_tools
from pyox_agents import Agent

agent = Agent(model="gpt-5-mini", tools=[fast_tools.count_words, fast_tools.most_common_words])
```

## Add a tool

In [`src/lib.rs`](src/lib.rs):

1. Write a struct with the tool's arguments. Doc comments become descriptions for the model.
2. Implement `RustTool` for it: a `NAME` and a `run` method.
3. Register it in the module with `add_tool::<YourTool>(module)?`.

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

The argument schema is generated from the struct, so you don't write it by hand.
Return `Err(message)` to tell the model something went wrong.

Rebuild after changes with `uv pip install ./examples/rust_tool_package` again
(inside this repo, `uv sync` does it for you).
