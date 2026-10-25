# Benchmarks

How much time and memory does the agent library itself cost? We compare pyox-agents with
[pydantic-ai](https://ai.pydantic.dev) on the same tasks, against the same fake model.

## Results

<!-- results:start -->
Python 3.13.17, pyox-agents 0.1.0, pydantic-ai 2.61.0, Linux x86_64 (Intel(R) Xeon(R) Processor @ 2.10GHz). 1000 runs per measurement, 100 at a time for throughput.

| Library | Startup | Memory after startup |
|---|--:|--:|
| pyox-agents | 73 ms | 23 MB |
| pydantic-ai | 1079 ms | 76 MB |

| Scenario | Library | Latency p50 | Latency p95 | CPU per run | Throughput | Peak memory |
|---|---|--:|--:|--:|--:|--:|
| text | pyox-agents | 56.4 ms | 69.0 ms | 65.9 ms | 23 runs/s | 56 MB |
| text | pydantic-ai | 137.6 ms | 174.5 ms | 137.8 ms | 7 runs/s | 117 MB |
| tool | pyox-agents | 4.3 ms | 5.9 ms | 4.1 ms | 434 runs/s | 33 MB |
| tool | pydantic-ai | 20.5 ms | 27.0 ms | 20.1 ms | 52 runs/s | 104 MB |
| parallel_tools | pyox-agents | 5.4 ms | 7.1 ms | 6.0 ms | 272 runs/s | 35 MB |
| parallel_tools | pydantic-ai | 31.3 ms | 42.8 ms | 31.3 ms | 31 runs/s | 107 MB |
<!-- results:end -->

- **Startup**: start Python, import the library and create an agent (one process each time).
- **Latency**: time of one complete run (all requests, streaming and tool calls), one run at a time.
- **CPU per run**: CPU time the process spends per run (all threads), one run at a time.
- **Throughput**: completed runs per second with 100 runs at the same time, in one process.
- **Peak memory**: highest memory (RSS) of the process during the whole measurement.

What the numbers say:

- With tools, pyox-agents is 5 to 6 times faster and serves 8 times more runs per second, with a third of the memory.
- Long streamed answers gain less (2.4 times faster): pyox-agents spends about 0.15 ms of CPU per streamed chunk,
  probably in handing each `TextDelta` from Rust to Python. Batching small deltas is worth trying next.
- pydantic-ai takes about 1 second to import; pyox-agents about 70 ms.

The fake model answers instantly, so these numbers are the overhead of each library. With a real model,
which takes seconds per answer, latencies are dominated by the model; the CPU and memory savings remain.

## What is measured

Each scenario is one agent run with streaming on, going through every event (as an app showing tokens live):

| Scenario         | What the fake model does                                                        | Requests |
|------------------|---------------------------------------------------------------------------------|---------:|
| `text`           | Streams a long answer (~1,800 characters in ~450 chunks of 4 characters)        | 1        |
| `tool`           | Asks for one tool call, then answers after getting the result                  | 2        |
| `parallel_tools` | Asks for five tool calls at once, then answers after getting the results      | 2        |

Both agents have the same system prompt and the same Python tool ([`agents.py`](agents.py)), and each run checks
that the final answer is the expected one, so a library can't look fast by skipping work.

## Tools

- **Fake model: [aimock](https://github.com/CopilotKit/aimock)** (`@copilotkit/aimock`, run with `npx`). A real HTTP
  server that speaks the OpenAI Chat Completions API, with streaming and tool calls. The answers are in
  [`fixtures.json`](fixtures.json). Matching is stateless (`hasToolResult`: has the conversation got a tool
  result?), so many runs can share the server at the same time. It costs about 0.4 ms per request, the same for
  both libraries.
- **Startup: [hyperfine](https://github.com/sharkdp/hyperfine)** (2.0 or newer). It runs a command many times and
  reports its mean time and memory, which is exactly the "start a new process" case.
- **Everything else: a small asyncio script** ([`measure.py`](measure.py)). Both libraries are *clients* used inside
  a Python program, so the most faithful load generator is a Python program running agents, measured with
  `time.perf_counter`, `time.process_time` and the process peak memory.

Why not the others:

- **wrk, locust, k6** load-test *HTTP servers*. To use them we would have to wrap each agent in a web server, and
  then we would also be measuring that web server. Our "server" is the fake model; the library under test is the
  client.
- **[llmposter](https://github.com/SkillDoAI/llmposter)** (a Rust mock server) was the first choice, but in
  version 0.5.0 every request on a reused connection waits ~40 ms (it doesn't disable Nagle's algorithm), which
  hides the libraries' own cost.
- **[ai-mock](https://pypi.org/project/ai-mock/)** streams one character per chunk, and as a Python server it
  could become the bottleneck under load.

To be fair to both libraries:

- Each library and scenario runs in its **own process**, so memory and caches never mix.
- Each measurement starts with 10 warm-up runs (connections opened, lazy imports done).
- Same Python, same machine, same fake model and the default settings of each library. For example, the tool is a
  plain (sync) function: pydantic-ai runs those in a thread pool, pyox-agents calls them directly.

## Run them

Requirements: [uv](https://docs.astral.sh/uv/), [Node.js](https://nodejs.org) (for `npx`), a Rust toolchain
(to build pyox-agents) and [hyperfine](https://github.com/sharkdp/hyperfine#installation) 2.0 or newer.
Linux or macOS.

```bash
cd benchmarks
uv run run_all.py              # everything; updates the results above (takes ~10 minutes)
uv run run_all.py --runs 100   # quicker, less precise
```

`run_all.py` starts the fake model, runs hyperfine and then `measure.py` for every library and scenario, and
writes the tables above and [`results.json`](results.json). To measure a single case by hand:

```bash
npx --yes --package=@copilotkit/aimock@1.46.0 llmock -f fixtures.json -p 4010 -c 4 &
uv run measure.py pyox-agents tool --runs 500
```

## Files

| File                               | What it does                                                    |
|------------------------------------|-----------------------------------------------------------------|
| [`agents.py`](agents.py)           | The same agent written with each library                        |
| [`fixtures.json`](fixtures.json)   | The answers of the fake model                                   |
| [`measure.py`](measure.py)         | Measures one library on one scenario and prints JSON            |
| [`run_all.py`](run_all.py)         | Runs everything and writes the results                         |

To add a library, add a `make_..._agent` function to `agents.py` and an entry in `LIBRARIES`
(and its package name in `PACKAGES` in `run_all.py`).
