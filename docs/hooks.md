# Hooks

Hooks are functions the agent calls at fixed points of every run, to watch it or change what happens next.
Return `None` to leave things as they are, or a new value to replace them:

| Hook                                   | When                               | Return a new...                         |
|----------------------------------------|------------------------------------|-----------------------------------------|
| `before_run(ctx)`                      | a run starts                       | -                                       |
| `before_model_request(ctx, request)`   | before each model request          | `ModelRequest`, for this request only   |
| `after_model_response(ctx, message)`   | after each model response          | assistant message                       |
| `before_tool_call(ctx, call)`          | before each tool call              | call (e.g. other `arguments`)           |
| `after_tool_call(ctx, call, result)`   | after each tool call               | result (`{"content", "is_error"}`)      |
| `after_run(ctx, result)`               | the run finished (`RunFinished`)   | -                                       |

`ctx` is the same [`RunContext`](tools.md#run-context-and-dependencies) tools get.

## Python hooks

```python
from pyox_agents import Agent, Hooks, RunContext, RunFinished, ToolCall


def no_deletes(ctx: RunContext, call: ToolCall) -> None:
    if call["name"] == "delete_file":
        raise PermissionError("Deleting files is not allowed")


class Timer:  # any object with hook methods works too
    def before_run(self, ctx: RunContext) -> None: ...
    async def after_run(self, ctx: RunContext, result: RunFinished) -> None: ...


agent = Agent(model="gpt-5-mini", tools=[...], hooks=[Hooks(before_tool_call=no_deletes), Timer()])
```

- Hooks can be sync or async, and run in the order given: each one sees the changes of the previous ones.
- Raising in `before_tool_call` or `after_tool_call` blocks the call: the model gets the error as the tool result.
  Raising in any other hook stops the run.

## Changing a request

A `ModelRequest` is a dict with what the next request sends: `instructions`, `messages`, `tools` (the names of
the tools offered, in order) and `model_parameters` (merged into the request body: `tool_choice`, `reasoning`,
`max_output_tokens`...). Change any of them for that request only; the history and the agent stay as they are,
so it is a safe place to trim long conversations, add reminders, or force a tool:

```python
from pyox_agents import ModelRequest


def search_first(ctx: RunContext, request: ModelRequest) -> ModelRequest | None:
    if ctx.turn == 1:  # the first request must call the search tool
        return {**request, "model_parameters": {**request["model_parameters"], "tool_choice": "required"}}
    return None
```

`tools` can only name the run's own tools (a `HookError` otherwise): drop some, or reorder them.

## Rust hooks

Register closures with `Hooks` from
[`pyox-tools`](https://github.com/daniprol/pyox-agents/blob/main/crates/pyox-tools/src/hooks.rs). The agent calls
them directly in Rust, without Python:

```rust
let guard = Hooks::new()
    .before_tool_call(|_context, call| {
        if call.name == "delete_file" {
            return Err("Deleting files is not allowed".into()); // the model sees this
        }
        Ok(())
    })
    .before_model_request(|_context, request| {
        // `request` is a `&mut ModelRequest`: change it in place, for this request only.
        request.model_parameters.insert("tool_choice".into(), "auto".into());
        Ok(())
    });
add_hooks(module, "guard", guard)?; // Python: Agent(hooks=[your_package.guard])
```

See `guard` in
[`examples/rust_tool_package`](https://github.com/daniprol/pyox-agents/blob/main/examples/rust_tool_package/src/lib.rs)
for a complete example, used by [`examples/rust_tools.py`](tools.md#your-own-rust-tools).
