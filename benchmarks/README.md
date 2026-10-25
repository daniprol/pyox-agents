# Benchmarks

How much time and memory does the agent library itself cost? We compare pyox-agents with
[pydantic-ai](https://ai.pydantic.dev) on the same tasks, against the same fake model, and then pyox-agents with
each way of grouping streamed text (`text_chunks`).

<!-- --8<-- [start:results] -->
## pyox-agents vs pydantic-ai

<!-- results:start -->
Python 3.13.17, pyox-agents 0.1.0, pydantic-ai 2.61.0, Linux x86_64 (Intel(R) Xeon(R) Processor @ 2.10GHz). 1000 runs per measurement, 100 at a time for throughput.

| Library | Startup | Memory after startup |
|---|--:|--:|
| pyox-agents | 83 ms | 24 MB |
| pydantic-ai | 1124 ms | 76 MB |

| Scenario | Library | First text | Latency p50 | Latency p95 | CPU per run | Throughput | Failed | Peak memory |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| text | pyox-agents | 2.5 ms | 63.3 ms | 84.8 ms | 74.2 ms | 21 runs/s | 0 | 61 MB |
| text | pydantic-ai | 7.3 ms | 124.5 ms | 153.5 ms | 122.6 ms | 7 runs/s | 0 | 119 MB |
| tool | pyox-agents | 2.3 ms | 5.1 ms | 8.3 ms | 4.8 ms | 354 runs/s | 0 | 34 MB |
| tool | pydantic-ai | 14.0 ms | 21.1 ms | 27.5 ms | 20.2 ms | 47 runs/s | 0 | 103 MB |
| parallel_tools | pyox-agents | 3.9 ms | 6.8 ms | 9.6 ms | 7.5 ms | 217 runs/s | 0 | 36 MB |
| parallel_tools | pydantic-ai | 23.7 ms | 30.8 ms | 42.4 ms | 30.8 ms | 30 runs/s | 0 | 108 MB |
<!-- results:end -->

- **Startup**: start Python, import the library and create an agent (one process each time).
- **First text**: time until the first piece of text reaches our code, one run at a time.
- **Latency**: time of one complete run (all requests, streaming and tool calls), one run at a time.
- **CPU per run**: CPU time the process spends per run (all threads), one run at a time.
- **Throughput**: completed runs per second with 100 runs at the same time, in one process.
- **Failed**: runs that raised an error during the throughput phase (they are not counted as completed).
- **Peak memory**: highest memory (RSS) of the process during the whole measurement.

What the numbers say:

- With tools, pyox-agents is 4 to 5 times faster and serves about 7 times more runs per second, with a third of
  the memory. Its first text also arrives about 6 times sooner.
- Long streamed answers gain less (2 times faster) with the default `text_chunks="delta"`: each `TextDelta`
  event costs pyox-agents about 0.15 ms of CPU, which adds up over ~450 events. See the streaming modes below.
- pydantic-ai takes about 1.1 seconds to import; pyox-agents about 80 ms.
- No run failed. Both clients resend a request that fails before any response arrives (pyox-agents:
  `Client(max_retries=2)`, the default), which covers keep-alive connections the server has just closed.

The fake model answers instantly, so these numbers are the overhead of each library. With a real model,
which takes seconds per answer, latencies are dominated by the model; the CPU and memory savings remain.

## Streaming modes (`text_chunks`)

pyox-agents can group streamed text into fewer, bigger `TextDelta` events (the `text_chunks` option). Here is what each
mode costs on the long streamed answer (5 markdown paragraphs, ~450 chunks from the API):

<!-- text-chunks:start -->
| Scenario | Mode | First text | Latency p50 | Latency p95 | CPU per run | Throughput | Failed | Peak memory |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| text | text_chunks='delta' | 2.3 ms | 64.5 ms | 85.2 ms | 75.7 ms | 20 runs/s | 0 | 61 MB |
| text | text_chunks=10 | 1.8 ms | 9.1 ms | 13.3 ms | 10.2 ms | 176 runs/s | 0 | 57 MB |
| text | text_chunks='paragraph' | 2.6 ms | 4.4 ms | 5.9 ms | 4.5 ms | 695 runs/s | 0 | 50 MB |
| text | text_chunks='message' | 4.3 ms | 4.9 ms | 6.3 ms | 4.0 ms | 678 runs/s | 0 | 48 MB |
<!-- text-chunks:end -->

What the numbers say:

- The cost is per event, not per character: grouping every 10 pieces makes the run 7 times faster and about 9 times
  more runs fit per second.
- `"paragraph"` is almost as cheap as `"message"` (5 events instead of 1).
- "First text" is misleading here because the fake model sends everything at once. With a real model (tens of
  milliseconds between pieces) the wait before the first `TextDelta` grows with the group size: about 10 pieces
  with `10` (a fraction of a second), one paragraph with `"paragraph"` (can be seconds), the whole answer with
  `"message"`. So `10` (or a similar number) is the safe choice for live UIs: most of the savings, almost no delay.

## Behind a FastAPI streaming endpoint

Each library serves `POST /chat` from one uvicorn process ([`server.py`](https://github.com/daniprol/pyox-agents/blob/main/benchmarks/server.py)) and streams the answer
text back. [k6](https://k6.io) sends requests from 1 to 100 concurrent users, each reading the whole answer
before sending the next request ([`load.js`](https://github.com/daniprol/pyox-agents/blob/main/benchmarks/load.js)).

<!-- server:start -->
Python 3.13.17, one uvicorn process, Intel(R) Xeon(R) Processor @ 2.10GHz. Each row: 20s of load.

| Scenario | Users | Library | Requests/s | First byte p50 | p99 | Latency p50 | p90 | p99 | Failed | Server CPU per request | Server peak memory |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| text | 1 | pyox-agents | 17 | 3.2 ms | 4.8 ms | 56.2 ms | 65.7 ms | 78.6 ms | 0 | 64.9 ms | 56 MB |
| text | 1 | pydantic-ai | 6 | 8.7 ms | 13.0 ms | 161.4 ms | 204.4 ms | 254.6 ms | 0 | 162.5 ms | 92 MB |
| text | 10 | pyox-agents | 23 | 4.5 ms | 13.7 ms | 439.1 ms | 463.3 ms | 481.8 ms | 0 | 70.4 ms | 59 MB |
| text | 10 | pydantic-ai | 6 | 49.4 ms | 134.8 ms | 1618.0 ms | 1798.6 ms | 1801.8 ms | 0 | 162.0 ms | 95 MB |
| text | 50 | pyox-agents | 24 | 18.1 ms | 100.3 ms | 2019.9 ms | 2200.5 ms | 2530.7 ms | 0 | 64.1 ms | 71 MB |
| text | 50 | pydantic-ai | 6 | 270.9 ms | 462.0 ms | 8708.3 ms | 8840.0 ms | 8952.2 ms | 0 | 172.5 ms | 109 MB |
| text | 100 | pyox-agents | 24 | 25.0 ms | 188.7 ms | 3949.8 ms | 5081.8 ms | 6700.9 ms | 0 | 64.2 ms | 86 MB |
| text | 100 | pydantic-ai | 5 | 451.1 ms | 1134.0 ms | 19971.4 ms | 20463.0 ms | 20505.9 ms | 0 | 191.7 ms | 120 MB |
| tool | 1 | pyox-agents | 192 | 2.7 ms | 4.8 ms | 4.9 ms | 6.0 ms | 8.5 ms | 0 | 4.5 ms | 55 MB |
| tool | 1 | pydantic-ai | 41 | 14.9 ms | 23.9 ms | 23.1 ms | 29.4 ms | 36.9 ms | 0 | 22.6 ms | 92 MB |
| tool | 10 | pyox-agents | 318 | 7.2 ms | 12.2 ms | 30.9 ms | 36.2 ms | 42.4 ms | 0 | 4.6 ms | 56 MB |
| tool | 10 | pydantic-ai | 44 | 118.3 ms | 239.9 ms | 222.0 ms | 270.2 ms | 332.3 ms | 0 | 22.5 ms | 98 MB |
| tool | 50 | pyox-agents | 346 | 47.4 ms | 94.2 ms | 142.3 ms | 172.8 ms | 203.8 ms | 0 | 4.1 ms | 61 MB |
| tool | 50 | pydantic-ai | 43 | 655.1 ms | 827.5 ms | 1145.4 ms | 1262.9 ms | 1293.9 ms | 0 | 23.0 ms | 111 MB |
| tool | 100 | pyox-agents | 339 | 101.6 ms | 249.6 ms | 264.7 ms | 478.6 ms | 628.8 ms | 0 | 4.2 ms | 65 MB |
| tool | 100 | pydantic-ai | 36 | 1487.4 ms | 2026.4 ms | 2707.2 ms | 2935.7 ms | 3395.7 ms | 0 | 27.5 ms | 122 MB |
<!-- server:end -->

- **Requests/s**: complete answers per second.
- **First byte**: until the first piece of the answer reaches the client. The server waits for the first text
  before starting the response, so this is "time to first text" as a user sees it.
- **Latency**: until the whole answer has arrived.
- **Server CPU per request** and **Server peak memory**: measured inside the server process.

What the numbers say:

- One uvicorn process is CPU-bound in both cases: throughput stops growing at about 10 users, and from there
  latency grows with the number of users (requests wait their turn). pyox-agents reaches that ceiling later:
  about 4 times more requests per second on long streamed answers (24 vs 6) and 8 to 9 times more with a tool
  call (340 vs 40).
- Time to first byte under load is where users feel it: with 100 users, the median wait before the answer starts
  is 25 ms vs 451 ms (streamed text) and 102 ms vs 1.5 s (tool call).
- With 100 users, the server process uses 65 to 86 MB with pyox-agents and about 120 MB with pydantic-ai.
- No request failed in either server.
- The server CPU per request is close to the library numbers above (within about 30%), so the agent library,
  not FastAPI or uvicorn, is most of the cost. To serve more, run more uvicorn workers or use a bigger `text_chunks`.

<!-- --8<-- [end:results] -->

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
- **Library overhead: a small asyncio script** ([`measure.py`](measure.py)). Both libraries are *clients* used inside
  a Python program, so the most faithful load generator is a Python program running agents, measured with
  `time.perf_counter`, `time.process_time` and the process peak memory.

- **FastAPI endpoint: [k6](https://k6.io)** (1.x). Here the thing under test *is* an HTTP server, so a load
  generator fits. k6 reports, per request, the time to first byte (`http_req_waiting`) and to the end of the
  streamed body (`http_req_duration`) with percentiles, checks every answer, and exports a JSON summary. It is a
  single Go binary, so it doesn't compete with the Python server for the GIL. *oha* and *wrk* are as fast but
  don't document a time-to-first-byte metric; *locust* is Python, and needs several worker processes to load a server
  without becoming the bottleneck itself.

Why not the others:

- **wrk, locust, k6** load-test *HTTP servers*. For `measure.py` we would have to wrap each agent in a web server,
  and then we would also be measuring that web server. The FastAPI benchmark does exactly that on purpose (see below).
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
(to build pyox-agents), [hyperfine](https://github.com/sharkdp/hyperfine#installation) 2.0 or newer and
[k6](https://grafana.com/docs/k6/latest/set-up/install-k6/) 1.x. Linux or macOS.

```bash
cd benchmarks
uv run run_all.py               # library benchmarks; updates the first two sections (~15 minutes)
uv run run_all.py --runs 100    # quicker, less precise
uv run load_test.py             # FastAPI benchmark; updates its section (~8 minutes)
uv run load_test.py --duration 5s
```

`run_all.py` starts the fake model, runs hyperfine and then `measure.py` for every library and scenario, and
writes the tables above and [`results.json`](results.json). To measure a single case by hand:

```bash
npx --yes --package=@copilotkit/aimock@1.46.0 llmock -f fixtures.json -p 4010 -c 4 &
uv run measure.py pyox-agents tool --runs 500

AGENT=pyox-agents uv run uvicorn server:app --port 8000 &
k6 run -e URL=http://127.0.0.1:8000/chat -e PROMPT="Tell me about agents." -e EXPECTED=Agents \
  -e VUS=10 -e SUMMARY_FILE=summary.json load.js
```

## Files

| File                               | What it does                                                    |
|------------------------------------|-----------------------------------------------------------------|
| [`agents.py`](agents.py)           | The same agent written with each library                        |
| [`fixtures.json`](fixtures.json)   | The answers of the fake model                                   |
| [`measure.py`](measure.py)         | Measures one agent on one scenario and prints JSON              |
| [`run_all.py`](run_all.py)         | Runs the library benchmarks and writes the results              |
| [`server.py`](server.py)           | FastAPI app streaming an agent's answer (`POST /chat`)          |
| [`load.js`](load.js)               | k6 script: users sending requests to the app in a loop          |
| [`load_test.py`](load_test.py)     | Runs the FastAPI benchmark and writes the results               |

To add a library, add a `make_..._agent` function to `agents.py` and an entry in `LIBRARIES`
(and its package name in `PACKAGES` in `run_all.py`). Both benchmarks pick it up.
