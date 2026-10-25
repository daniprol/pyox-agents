# fast-history: a conversation history store written in Rust

A minimal Python package with a history store written in Rust: it keeps
conversations in SQLite through [sqlx](https://github.com/launchbadge/sqlx).
Copy this folder to start your own (Postgres, MySQL, or any Rust ORM).

## Use it

```bash
uv pip install ./examples/rust_history_store    # compile and install (needs Rust)
```

```python
import fast_history
from pyox_agents import Agent

# Once, at startup: opens the database and its connection pool.
store = fast_history.sqlite_history("sqlite://chats.db", max_connections=8)
agent = Agent(model="gpt-5-mini", history_store=store)

# Each run names its conversation: the agent loads it before the run and saves the new messages after.
await agent.run("Hi, I'm Dani", conversation_id="user-42")
await agent.run("What's my name?", conversation_id="user-42")

store.load("user-42")  # the messages as dicts, e.g. to show them in a UI
```

The agent calls `load` and `save` directly in Rust: no Python runs and the
messages never become Python objects.

## Write your own

In [`src/lib.rs`](src/lib.rs):

1. A struct holding your connection pool.
2. Implement `HistoryStore` for it: `load` returns a conversation's messages,
   oldest first; `save` appends the messages a run added. Both are `async`.
3. A `#[pyfunction]` that creates the pool (inside `runtime()`) and returns
   `RustHistoryStore::new(your_store)`.

```rust
struct SqliteHistory {
    pool: SqlitePool,
}

impl HistoryStore for SqliteHistory {
    async fn load(&self, conversation_id: &str) -> Result<Vec<Message>, String> { ... }
    async fn save(&self, conversation_id: &str, new_messages: &[Message]) -> Result<(), String> { ... }
}
```

`Message` is the same type pyox-agents uses (from `pyox_tools::messages`).
Return `Err(message)` when something fails: the run raises `HistoryError`.

## In production

- Create the store once per process and share it: every run of every agent
  uses the same pool. Size the pool to about the number of runs that load at
  the same time; loads are one short query.
- One row per message keeps saves cheap: a run only appends what it added.
- For Postgres: `PgPool`, `$1` placeholders, and a `JSONB` column for the message.

Rebuild after changes with `uv pip install ./examples/rust_history_store` again
(inside this repo, `uv sync` does it for you).
