# Notes for coding agents

pyox-agents is a minimal, fast Python library for AI agents with a Rust core.
Keep it small, clean and readable. Do not overengineer.

## The main rule: Rust that any Python developer can read

Rust code and its comments must be easy to understand for a Python developer
who has never used Rust.

- Use clear, plain names (`history`, `tool_calls`, `request_body`), not
  abbreviations or Rust jargon.
- Prefer simple, straight-line code over clever generics, macros, traits or
  lifetimes. Use them only when they clearly make the code simpler.
- Comments explain *why*, and briefly explain any Rust feature a Python
  developer would not know (for example `#[serde(untagged)]`, `Option`, `?`,
  `Arc`), ideally by comparing it to Python.
- The same applies to the Python code: small functions, clear names, no magic.

## Design

- Rust owns HTTP, streaming, JSON parsing and the agent loop. Python defines
  agents and tools.
- The history format is our own (`crates/pyox-tools/src/messages.rs`). Each API
  has one adapter file (for example `src/openai_chat.rs`) that translates it.
  The agent loop does not know which API it talks to.
- The Decisions API (`src/openai_decisions.rs`) is not a model turn: it is
  called directly with `decide(...)`, and reuses the same `Client` and the
  Responses API's input items.

## Workflow

- Tools: uv, ruff, ty, pytest, cargo fmt, clippy; hooks run with prek
  (`prek.toml`). `uv run pytest` builds the extension and runs the tests.
- Tests use the local fake OpenAI server and reusable fixtures and factories.
  Avoid mocking or patching unless there is no other way.
