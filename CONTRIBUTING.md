# Contributing

Thanks for helping. pyox-agents aims to stay small, fast and readable, so the bar for new code is that a Python
developer who has never written Rust can follow it. [`AGENTS.md`](AGENTS.md) has the style rules (for people and
coding agents alike).

Every change goes through a pull request into `main`, with CI passing. Requires [uv](https://docs.astral.sh/uv/)
and a Rust toolchain ([rustup](https://rustup.rs/)).

## Setup

```bash
uv sync                       # build the Rust extension and install the dev tools
uv run prek install           # install the git hooks (once)
```

Git hooks ([prek](https://prek.j178.dev), configured in [`prek.toml`](https://github.com/daniprol/pyox-agents/blob/main/prek.toml)):

| When       | What runs                                                         |
|------------|-------------------------------------------------------------------|
| `git commit` | `ruff check --fix`, `ruff format`, `cargo fmt`, basic file checks |
| `git push`   | `ty check`, `cargo clippy`, `cargo test`, `pytest`              |

Run them by hand with `uv run prek run --all-files` (add `--stage pre-push` for the push hooks).

Tests use a local fake OpenAI server. A few live tests in [`tests/test_live.py`](https://github.com/daniprol/pyox-agents/blob/main/tests/test_live.py) call a
real model on Azure OpenAI instead; they are skipped by default. Run them with
`uv run pytest -m live` after setting `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and optionally
`LIVE_MODEL` (the deployment, `gpt-6-luna` by default). CI runs them on pushes to `main` and `dev`, from
the `live` GitHub environment, with `LIVE_TEST_WORKERS` (in `ci.yml`) tests at a time.

The documentation site lives in [`docs/`](https://github.com/daniprol/pyox-agents/blob/main/docs) and [`zensical.toml`](https://github.com/daniprol/pyox-agents/blob/main/zensical.toml), built with
[Zensical](https://zensical.org). Preview it with `uv run --only-group docs zensical serve`; every push to `main`
publishes it to GitHub Pages ([`.github/workflows/docs.yml`](https://github.com/daniprol/pyox-agents/blob/main/.github/workflows/docs.yml)).

CI ([`.github/workflows/ci.yml`](https://github.com/daniprol/pyox-agents/blob/main/.github/workflows/ci.yml)) runs both hook stages on Python 3.11 and 3.14, plus a
coverage job. The coverage report (Python and Rust, including the Rust code exercised by the Python tests) is
shown in the run summary and uploaded as an artifact.

Releases go to [PyPI](https://pypi.org/p/pyox-agents). The version lives only in `pyproject.toml`. To release
from `main`, bump it, commit, and push a matching tag:

```bash
uv version --bump patch  # fixes and features; --bump minor for breaking changes (before 1.0)
git commit -am "Release $(uv version --short)" && git push
git tag "v$(uv version --short)" && git push origin "v$(uv version --short)"
```

The tag starts [`.github/workflows/release.yml`](https://github.com/daniprol/pyox-agents/blob/main/.github/workflows/release.yml):
it reruns CI on the tagged commit, builds the wheels and the sdist, waits for approval in the `pypi` environment,
publishes with trusted publishing (no API tokens) and creates a GitHub release. Running the workflow by hand
publishes to TestPyPI instead.

Tests talk to a real local HTTP server that imitates the OpenAI API ([`tests/fake_openai.py`](https://github.com/daniprol/pyox-agents/blob/main/tests/fake_openai.py)),
so there is no mocking. Build responses with the factories in [`tests/factories.py`](https://github.com/daniprol/pyox-agents/blob/main/tests/factories.py)
and create agents with the `make_agent` fixture:

```python
async def test_example(fake_openai, make_agent):
    fake_openai.queue(tool_calls_response(tool_call("add", a=1, b=2)), text_response("3"))
    result = await make_agent(tools=[add]).run("1 + 2?")
    assert result.text == "3"
```

When you add or change a feature, update its page in [`docs/`](docs) too, and the
[agent skill](skills/pyox-agents/SKILL.md) if it changes how the library is used.

## Code map

```text
Python                               Rust (src/)
------                               -----------
Agent(...)  ───────────────────────► Engine         config shared by all runs (engine.rs)
agent.execute("...")  ─────────────► Run            one run: history + event queue (run.rs)
async for event in run  ◄─ events ── agent loop     request → stream → tools → repeat (agent_loop.rs)
                                       ├── model_api.rs     picks the API adapter
                                       ├── openai_chat.rs   Chat Completions: request body + chunk parsing
                                       ├── openai_responses.rs  Responses: request body + event parsing
                                       ├── sse.rs           server-sent events decoding
                                       ├── ws.rs, ws_pool.rs  Responses over WebSocket: one connection, the pool
                                       └── tool_call.rs     one tool call: hooks, approval, the tool, failures
                                             └── tools/     Python tools are awaited on Python's event
                                                            loop; Rust tools (from your own package)
                                                            run on a Rust thread
run.ai_sdk(), run.ag_ui(...)  ◄─ text ── run.rs   events in batches → ai_sdk.rs, ag_ui.rs
```

The Rust code is written to be read by Python developers: small files, descriptive names and comments
explaining the Rust-specific parts. Start with [`src/lib.rs`](https://github.com/daniprol/pyox-agents/blob/main/src/lib.rs).
