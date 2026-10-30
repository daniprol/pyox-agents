# Examples

Run all commands from the root of the repository.

## 1. Setup (once)

```bash
uv sync                               # builds pyox-agents and the example Rust packages
export OPENAI_API_KEY=sk-...
```

## 2. An agent with a Python tool and a built-in Rust tool

```bash
uv run examples/basic.py
```

Streams the answer, shows each tool call, and writes `weather.txt`.

## 3. An agent with your own Rust tools

```bash
uv run examples/rust_tools.py
```

Uses `count_words` and `most_common_words` from [`rust_tool_package`](rust_tool_package),
a separate Rust package, plus its `guard` hooks. See its README to write your own.

## 4. Conversations stored in a database

```bash
uv run examples/python_history.py     # a Python store using the SQLAlchemy ORM
uv run examples/rust_history.py       # a Rust store using sqlx: loading never touches Python
```

Run each twice: the second time, the agent remembers the first conversation. Both keep conversations in
SQLite, one row per message. The Rust one comes from [`rust_history_store`](rust_history_store), a separate
Rust package; see its README to write your own.

## 5. Images and PDFs

```bash
uv run examples/images_and_files.py path/to/some.pdf
```

Asks about a photo (by web address) and summarizes the PDF (sent inline).

## 6. Tools that wait for a person

```bash
uv run examples/human_in_the_loop.py
```

A tool asks you a question (the run stops and resumes with your answer), then a tool waits for your approval
(answered live, while the run waits).

## 7. A web server for AI SDK and AG-UI frontends

```bash
uv run uvicorn examples.web_server:app --reload
```

Serves the agent of example 6 to `useChat()` (AI SDK) at `/api/chat` and to AG-UI clients (CopilotKit) at
`/api/agent`. See [Web frontends](https://daniprol.github.io/pyox-agents/web-frontends/).

## Using another OpenAI-compatible server

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1   # e.g. Ollama
```

Then change `model=` in the example to a model that server has.
