# Benchmarks

How much time and memory does the agent library itself cost? We compare pyox-agents with
[pydantic-ai](https://ai.pydantic.dev) on the same tasks, against the same fake model, and then pyox-agents with
each way of grouping streamed text (`text_chunks`).

## pyox-agents vs pydantic-ai

<!-- results:start -->
Python 3.13.17, pyox-agents 0.1.0, pydantic-ai 2.61.0, Linux x86_64 (Intel(R) Xeon(R) Processor @ 2.10GHz). 1000 runs per measurement, 100 at a time for throughput.

| Library | Startup | Memory after startup |
|---|--:|--:|
| pyox-agents | 77 ms | 23 MB |
| pydantic-ai | 1044 ms | 76 MB |

| Scenario | Library | First text | Latency p50 | Latency p95 | CPU per run | Throughput | Failed | Peak memory |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| text | pyox-agents | 2.3 ms | 56.5 ms | 71.4 ms | 67.0 ms | 24 runs/s | 0 | 59 MB |
| text | pydantic-ai | 7.2 ms | 143.3 ms | 195.1 ms | 145.7 ms | 6 runs/s | 0 | 117 MB |
| tool | pyox-agents | 2.0 ms | 4.7 ms | 6.6 ms | 4.4 ms | 395 runs/s | 0 | 34 MB |
| tool | pydantic-ai | 14.4 ms | 22.4 ms | 31.2 ms | 22.4 ms | 43 runs/s | 0 | 103 MB |
| parallel_tools | pyox-agents | 3.5 ms | 6.1 ms | 8.1 ms | 6.7 ms | 239 runs/s | 0 | 36 MB |
| parallel_tools | pydantic-ai | 25.4 ms | 33.6 ms | 45.5 ms | 33.7 ms | 29 runs/s | 0 | 108 MB |
<!-- results:end -->

- **Startup**: start Python, import the library and create an agent (one process each time).
- **First text**: time until the first piece of text reaches our code, one run at a time.
- **Latency**: time of one complete run (all requests, streaming and tool calls), one run at a time.
- **CPU per run**: CPU time the process spends per run (all threads), one run at a time.
- **Throughput**: completed runs per second with 100 runs at the same time, in one process.
- **Failed**: runs that raised an error during the throughput phase (they are not counted as completed).
- **Peak memory**: highest memory (RSS) of the process during the whole measurement.

What the numbers say:

- With tools, pyox-agents is 5 times faster and serves 8 to 9 times more runs per second, with a third of the
  memory. Its first text also arrives about 7 times sooner.
- Long streamed answers gain less (2.5 times faster) with the default `text_chunks="delta"`: each `TextDelta`
  event costs pyox-agents about 0.13 ms of CPU, which adds up over ~450 events. See the streaming modes below.
- pydantic-ai takes about 1 second to import; pyox-agents about 75 ms.
- Under 100 concurrent streams, pyox-agents occasionally fails a run with `NetworkError` ("error sending request",
  about 1 in 1,000 runs, not always reproducible). pydantic-ai's OpenAI client retries such errors twice by default,
  so they would not show there.

The fake model answers instantly, so these numbers are the overhead of each library. With a real model,
which takes seconds per answer, latencies are dominated by the model; the CPU and memory savings remain.

## Streaming modes (`text_chunks`)

pyox-agents can group streamed text into fewer, bigger `TextDelta` events (see the main README). Here is what each
mode costs on the long streamed answer (5 markdown paragraphs, ~450 chunks from the API):

<!-- text-chunks:start -->
| Scenario | Mode | First text | Latency p50 | Latency p95 | CPU per run | Throughput | Failed | Peak memory |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| text | text_chunks='delta' | 2.0 ms | 56.0 ms | 67.8 ms | 65.7 ms | 24 runs/s | 1 | 58 MB |
| text | text_chunks=10 | 1.6 ms | 7.6 ms | 9.9 ms | 8.8 ms | 211 runs/s | 0 | 56 MB |
| text | text_chunks='paragraph' | 2.6 ms | 4.5 ms | 5.8 ms | 4.6 ms | 740 runs/s | 0 | 51 MB |
| text | text_chunks='message' | 4.3 ms | 4.7 ms | 6.0 ms | 4.1 ms | 746 runs/s | 0 | 47 MB |
<!-- text-chunks:end -->

What the numbers say:

- The cost is per event, not per character: grouping every 10 pieces makes the run 7 times faster and about 9 times
  more runs fit per second.
- `"paragraph"` is almost as cheap as `"message"` (5 events instead of 1).
- "First text" is misleading here because the fake model sends everything at once. With a real model (tens of
  milliseconds between pieces) the wait before the first `TextDelta` grows with the group size: about 10 pieces
  with `10` (a fraction of a second), one paragraph with `"paragraph"` (can be seconds), the whole answer with
  `"message"`. So `10` (or a similar number) is the safe choice for live UIs: most of the savings, almost no delay.

## What is measured

Each scenario is one agent run with streaming on, going through every event (as an app showing tokens live):

| Scenario         | What the fake model does                                                        | Requests |
|------------------|---------------------------------------------------------------------------------|---------:|
| `text`           | Streams a long answer (5 paragraphs, ~1,800 characters in ~450 chunks)          | 1        |
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
