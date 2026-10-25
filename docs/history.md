# History

Messages are plain dicts (typed as `UserMessage`, `AssistantMessage` and `ToolMessage`). The same format goes in
and out:

```python
{"role": "user", "content": "Hi"}
{"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image", "url": "https://..."}]}
{"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "add", "arguments": {"a": 1}}]}
{"role": "tool", "tool_call_id": "call_1", "name": "add", "content": "3", "is_error": False}
```

## Continue a conversation

Pass the messages of a previous run as `history=`:

```python
first = await agent.run("I am Dani")
second = await agent.run("What is my name?", history=first.messages)
```

`history=` also takes JSON text (`str` or `bytes`), which is parsed in Rust without building Python dicts. Use it
when the messages come from a database:

```python
await agent.run("What is my name?", history=first.messages_json())
```

`RunFinished.new_messages` (and `new_messages_json()`) hold only what the run added, to append to your storage.

## History stores

Or let the agent load and save conversations for you. Give it a history store once, and name the conversation on
each run:

```python
agent = Agent(model="gpt-5-mini", history_store=store)
await agent.run("Hi again", conversation_id="user-42")
```

Before the run, the agent calls `store.load(conversation_id)`; after it, `store.save(conversation_id,
new_messages_json)`. Failed runs are not saved. If the store fails or returns something invalid, the run raises
(`HistoryError` for invalid data).

=== "Python"

    Any object with a `load` method (and optionally `save`), sync or async. A plain `load(conversation_id)`
    function works too. `load` returns message dicts, or JSON text (faster: parsed in Rust).

    ```python
    --8<-- "examples/python_history.py"
    ```

=== "Rust"

    Implement the `HistoryStore` trait from
    [`pyox-tools`](https://github.com/daniprol/pyox-agents/tree/main/crates/pyox-tools) on a struct holding
    your connection pool (sqlx, diesel, SeaORM...). The agent calls it directly in Rust, so loading never goes
    through Python:

    ```rust
    struct SqliteHistory {
        pool: SqlitePool,
    }

    impl HistoryStore for SqliteHistory {
        async fn load(&self, conversation_id: &str) -> Result<Vec<Message>, String> { ... }
        async fn save(&self, conversation_id: &str, new_messages: &[Message]) -> Result<(), String> { ... }
    }
    ```

    Expose it with a `#[pyfunction]` that returns `RustHistoryStore::new(your_store)`, and use it from Python:

    ```python
    --8<-- "examples/rust_history.py"
    ```

    [`examples/rust_history_store`](https://github.com/daniprol/pyox-agents/tree/main/examples/rust_history_store)
    is a complete SQLite store to copy.

Try both from the repository root (run each twice: the second time, the agent remembers):

```bash
uv sync
uv run examples/python_history.py
uv run examples/rust_history.py
```

!!! tip "In production"
    Create the store once per process and share it between agents and runs: it owns the connection pool.
    Keep one row per message, so saving a run only appends what it added.
