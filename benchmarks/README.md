# Benchmarks

How much time and memory does the agent library itself cost? We run the same agent, with the same tool and the
same streamed answers, in pyox-agents and seven popular Python agent frameworks, against the same fake OpenAI
server: [pydantic-ai](https://ai.pydantic.dev),
the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/),
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework),
[LangGraph](https://langchain-ai.github.io/langgraph/), [LangChain](https://docs.langchain.com/oss/python/langchain/overview),
[LlamaIndex](https://developers.llamaindex.ai/python/framework/) and [Agno](https://docs.agno.com).
The fake server answers instantly, so the numbers are the overhead of each library: called directly from Python,
with each pyox-agents streaming mode, and behind a FastAPI streaming endpoint under load.

## Quickstart

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), [Node.js](https://nodejs.org) 18+ (the
fake server runs with `npx`), a [Rust toolchain](https://rustup.rs) (to build pyox-agents),
[hyperfine](https://github.com/sharkdp/hyperfine#installation) 2.0+ and
[k6](https://grafana.com/docs/k6/latest/set-up/install-k6/) 1.x, on Linux or macOS.

```bash
git clone https://github.com/daniprol/pyox-agents
cd pyox-agents/benchmarks
uv sync                                     # installs every library and builds pyox-agents

uv run run_all.py --runs 100                # quick library benchmark of all libraries (~10 minutes)
uv run run_all.py --only langgraph agno     # only these libraries, compared with pyox-agents
uv run load_test.py --duration 5s           # quick FastAPI benchmark (~10 minutes)
```

`--only` prints the tables and leaves this README alone. Without it, the scripts rewrite the tables below and
`results.json` / `load_test_results.json`. The full runs (`uv run run_all.py`, `uv run load_test.py`)
take about an hour each.

<!-- --8<-- [start:results] -->
## Called directly from Python

<!-- results:start -->
Python 3.13.17, Linux x86_64 (Intel(R) Xeon(R) Processor @ 2.80GHz). 500 runs per measurement, 100 at a time for throughput. Versions: pyox-agents 0.1.0, pydantic-ai 2.61.0, openai-agents 0.20.0, agent-framework 1.23.0, langgraph 1.2.18, langchain 1.4.7, llama-index 0.14.25, agno 3.1.12.

### Overview

Latency p50 of one run, fastest first; in parentheses, how many times the pyox-agents time it is. Lower is better.

| Library | Startup | text | tool | parallel_tools |
|---|--:|--:|--:|--:|
| pyox-agents | 132 ms | 105.2 ms (1.0x) | 8.3 ms (1.0x) | 10.9 ms (1.0x) |
| agno | 1614 ms | 121.2 ms (1.2x) | 17.6 ms (2.1x) | 28.1 ms (2.6x) |
| agent-framework | 1462 ms | 147.7 ms (1.4x) | 21.7 ms (2.6x) | 32.6 ms (3.0x) |
| openai-agents | 3176 ms | 111.8 ms (1.1x) | 24.7 ms (3.0x) | 35.4 ms (3.3x) |
| langchain | 2760 ms | 151.1 ms (1.4x) | 27.2 ms (3.3x) | 44.1 ms (4.0x) |
| langgraph | 2752 ms | 158.6 ms (1.5x) | 27.6 ms (3.3x) | 46.2 ms (4.2x) |
| pydantic-ai | 2158 ms | 213.1 ms (2.0x) | 35.4 ms (4.2x) | 48.4 ms (4.4x) |
| llama-index | 3243 ms | 181.5 ms (1.7x) | 41.0 ms (4.9x) | 65.8 ms (6.0x) |

### `text`: one long streamed answer

| Library | Latency p50 | Latency p95 | First text | CPU per run | Throughput | Failed | Peak memory |
|---|--:|--:|--:|--:|--:|--:|--:|
| pyox-agents | 105.2 ms | 135.9 ms | 3.3 ms | 120.0 ms | 14 runs/s | 0 | 50 MB |
| openai-agents | 111.8 ms | 157.3 ms | 74.4 ms | 113.6 ms | 8 runs/s | 0 | 184 MB |
| agno | 121.2 ms | 197.2 ms | 8.2 ms | 133.3 ms | 9 runs/s | 0 | 100 MB |
| agent-framework | 147.7 ms | 212.8 ms | 7.6 ms | 157.2 ms | 6 runs/s | 0 | 381 MB |
| langchain | 151.1 ms | 230.3 ms | 148.9 ms | 160.0 ms | 6 runs/s | 0 | 234 MB |
| langgraph | 158.6 ms | 240.4 ms | 155.5 ms | 166.6 ms | 6 runs/s | 0 | 223 MB |
| llama-index | 181.5 ms | 264.8 ms | 170.3 ms | 191.4 ms | 4 runs/s | 0 | 282 MB |
| pydantic-ai | 213.1 ms | 266.1 ms | 10.0 ms | 215.2 ms | 4 runs/s | 0 | 126 MB |

### `tool`: one tool call, then the answer

| Library | Latency p50 | Latency p95 | First text | CPU per run | Throughput | Failed | Peak memory |
|---|--:|--:|--:|--:|--:|--:|--:|
| pyox-agents | 8.3 ms | 11.7 ms | 3.5 ms | 7.3 ms | 230 runs/s | 0 | 35 MB |
| agno | 17.6 ms | 26.1 ms | 12.8 ms | 17.3 ms | 57 runs/s | 0 | 94 MB |
| agent-framework | 21.7 ms | 31.4 ms | 15.1 ms | 21.6 ms | 40 runs/s | 0 | 129 MB |
| openai-agents | 24.7 ms | 38.3 ms | 22.7 ms | 25.2 ms | 41 runs/s | 0 | 143 MB |
| langchain | 27.2 ms | 39.8 ms | 26.0 ms | 26.9 ms | 40 runs/s | 0 | 125 MB |
| langgraph | 27.6 ms | 40.0 ms | 26.0 ms | 27.1 ms | 37 runs/s | 0 | 123 MB |
| pydantic-ai | 35.4 ms | 52.8 ms | 23.3 ms | 35.9 ms | 25 runs/s | 0 | 119 MB |
| llama-index | 41.0 ms | 62.5 ms | 36.9 ms | 42.2 ms | 23 runs/s | 0 | 196 MB |

### `parallel_tools`: five tool calls at once, then the answer

| Library | Latency p50 | Latency p95 | First text | CPU per run | Throughput | Failed | Peak memory |
|---|--:|--:|--:|--:|--:|--:|--:|
| pyox-agents | 10.9 ms | 15.3 ms | 6.1 ms | 11.5 ms | 147 runs/s | 0 | 36 MB |
| agno | 28.1 ms | 40.2 ms | 22.8 ms | 28.2 ms | 37 runs/s | 0 | 99 MB |
| agent-framework | 32.6 ms | 51.3 ms | 25.0 ms | 34.5 ms | 26 runs/s | 0 | 157 MB |
| openai-agents | 35.4 ms | 53.9 ms | 33.0 ms | 36.0 ms | 29 runs/s | 0 | 147 MB |
| langchain | 44.1 ms | 60.0 ms | 42.8 ms | 43.3 ms | 25 runs/s | 0 | 133 MB |
| langgraph | 46.2 ms | 59.9 ms | 44.3 ms | 44.6 ms | 24 runs/s | 0 | 128 MB |
| pydantic-ai | 48.4 ms | 68.2 ms | 37.0 ms | 48.9 ms | 18 runs/s | 0 | 122 MB |
| llama-index | 65.8 ms | 91.2 ms | 61.6 ms | 66.4 ms | 14 runs/s | 0 | 204 MB |

### Startup

| Library | Startup | Memory after startup |
|---|--:|--:|
| pyox-agents | 132 ms | 25 MB |
| agent-framework | 1462 ms | 67 MB |
| agno | 1614 ms | 74 MB |
| pydantic-ai | 2158 ms | 92 MB |
| langgraph | 2752 ms | 108 MB |
| langchain | 2760 ms | 109 MB |
| openai-agents | 3176 ms | 114 MB |
| llama-index | 3243 ms | 132 MB |
<!-- results:end -->

- **Startup**: start Python, import the library and create an agent (one process each time).
- **First text**: time until the first piece of text reaches our code, one run at a time.
- **Latency**: time of one complete run (all requests, streaming and tool calls), one run at a time.
- **CPU per run**: CPU time the process spends per run (all threads), one run at a time.
- **Throughput**: completed runs per second with 100 runs at the same time, in one process.
- **Failed**: runs that raised an error during the throughput phase (they are not counted as completed).
- **Peak memory**: highest memory (RSS) of the process during the whole measurement.

What the numbers say:

- With tools, pyox-agents is 2 to 6 times faster than every other library (2.1x Agno, 3.3x LangChain, 4.2x
  pydantic-ai, 4.9x LlamaIndex on one tool call), serves 4 to 10 times more runs per second, and uses a third to a
  fifth of the memory.
- Long streamed answers gain less (1.1 to 2 times faster) with the default `text_chunks="delta"`: each
  `TextDelta` event costs pyox-agents about 0.25 ms of CPU, which adds up over ~450 events. With `text_chunks=10`
  the same answer takes 15 ms, 7 to 14 times less than the others. See the streaming modes below.
- Startup: about 0.1 s for pyox-agents, 1.5 to 3.2 s for the others (mostly importing the library).
- "First text" in `text` is close to the whole run for the OpenAI Agents SDK, LangChain, LangGraph and LlamaIndex.
  They read the model's stream in a separate asyncio task and pass the pieces through a queue, so with this fake
  model, which sends the whole answer at once, our code most likely only gets its turn once the whole answer has
  been read. With a real model (tens of milliseconds between pieces), they stream text as it arrives.
- No run failed. The clients resend a request that fails before any response arrives (pyox-agents:
  `Client(max_retries=2)`, the default), which covers keep-alive connections the server has just closed.

The fake model answers instantly, so these numbers are the overhead of each library. With a real model,
which takes seconds per answer, latencies are dominated by the model; the CPU and memory savings remain.

## Streaming modes (`text_chunks`)

pyox-agents can group streamed text into fewer, bigger `TextDelta` events (the `text_chunks` option). Here is what each
mode costs on the long streamed answer (5 markdown paragraphs, ~450 chunks from the API):

<!-- text-chunks:start -->
| Mode | Latency p50 | Latency p95 | First text | CPU per run | Throughput | Failed | Peak memory |
|---|--:|--:|--:|--:|--:|--:|--:|
| text_chunks='paragraph' | 6.7 ms | 8.6 ms | 4.0 ms | 6.4 ms | 432 runs/s | 0 | 48 MB |
| text_chunks='message' | 7.6 ms | 9.8 ms | 6.8 ms | 6.1 ms | 409 runs/s | 0 | 46 MB |
| text_chunks=10 | 14.7 ms | 21.3 ms | 3.2 ms | 15.5 ms | 112 runs/s | 0 | 57 MB |
| text_chunks='delta' | 111.1 ms | 142.1 ms | 3.3 ms | 125.0 ms | 12 runs/s | 0 | 49 MB |
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
Python 3.13.17, one uvicorn process, Intel(R) Xeon(R) Processor @ 2.80GHz. Each measurement: 20s of load.

### `text`: one long streamed answer

| Library | Req/s, 1 user | Req/s, 10 users | Req/s, 50 users | Req/s, 100 users | First byte p50, 100 users | Latency p50, 100 users | CPU per request | Peak memory | Failed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pyox-agents | 10 | 15 | 16 | 15 | 61 ms | 6049 ms | 101.7 ms | 80 MB | 0 |
| openai-agents | 8 | 8 | 8 | 8 | 6382 ms | 11495 ms | 119.9 ms | 167 MB | 0 |
| agno | 6 | 6 | 6 | 7 | 5138 ms | 5893 ms | 148.9 ms | 102 MB | 0 |
| langchain | 6 | 7 | 6 | 6 | 6077 ms | 12367 ms | 158.2 ms | 148 MB | 0 |
| langgraph | 6 | 6 | 6 | 6 | 5692 ms | 14081 ms | 161.1 ms | 143 MB | 0 |
| llama-index | 6 | 5 | 5 | 5 | 13359 ms | 18370 ms | 195.3 ms | 279 MB | 0 |
| agent-framework | 5 | 5 | 5 | 5 | 6251 ms | 6996 ms | 197.3 ms | 175 MB | 2 |
| pydantic-ai | 4 | 4 | 3 | 3 | 583 ms | 28790 ms | 287.5 ms | 135 MB | 0 |

### `tool`: one tool call, then the answer

| Library | Req/s, 1 user | Req/s, 10 users | Req/s, 50 users | Req/s, 100 users | First byte p50, 100 users | Latency p50, 100 users | CPU per request | Peak memory | Failed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pyox-agents | 142 | 222 | 246 | 220 | 138 ms | 403 ms | 6.4 ms | 65 MB | 0 |
| agno | 44 | 53 | 49 | 51 | 1902 ms | 1909 ms | 19.5 ms | 109 MB | 0 |
| openai-agents | 43 | 45 | 47 | 40 | 1884 ms | 2444 ms | 24.9 ms | 159 MB | 0 |
| agent-framework | 41 | 44 | 43 | 37 | 2481 ms | 2494 ms | 26.9 ms | 148 MB | 0 |
| langchain | 36 | 46 | 41 | 35 | 2059 ms | 2814 ms | 28.1 ms | 132 MB | 0 |
| langgraph | 36 | 41 | 39 | 34 | 2168 ms | 2759 ms | 29.2 ms | 136 MB | 0 |
| llama-index | 25 | 25 | 22 | 23 | 3562 ms | 4303 ms | 42.9 ms | 209 MB | 0 |
| pydantic-ai | 28 | 30 | 27 | 22 | 2312 ms | 4572 ms | 45.2 ms | 134 MB | 0 |

<details><summary>All the numbers</summary>

| Scenario | Library | Users | Requests/s | First byte p50 | p99 | Latency p50 | p90 | p99 | Failed | Server CPU per request | Server peak memory |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| text | pyox-agents | 1 | 10 | 4.3 ms | 7.5 ms | 93.6 ms | 111.4 ms | 124.7 ms | 0 | 106.1 ms | 57 MB |
| text | pydantic-ai | 1 | 4 | 11.6 ms | 18.6 ms | 249.5 ms | 286.5 ms | 324.0 ms | 0 | 249.1 ms | 106 MB |
| text | openai-agents | 1 | 8 | 66.4 ms | 133.0 ms | 120.5 ms | 152.8 ms | 201.2 ms | 0 | 123.2 ms | 136 MB |
| text | agent-framework | 1 | 5 | 8.1 ms | 14.4 ms | 183.4 ms | 218.5 ms | 437.9 ms | 0 | 191.9 ms | 120 MB |
| text | langgraph | 1 | 6 | 140.3 ms | 257.9 ms | 152.5 ms | 190.9 ms | 269.1 ms | 0 | 155.6 ms | 122 MB |
| text | langchain | 1 | 6 | 139.7 ms | 299.3 ms | 151.8 ms | 195.3 ms | 308.6 ms | 0 | 159.6 ms | 124 MB |
| text | llama-index | 1 | 6 | 154.7 ms | 315.0 ms | 171.2 ms | 209.2 ms | 330.9 ms | 0 | 176.8 ms | 190 MB |
| text | agno | 1 | 6 | 8.3 ms | 11.6 ms | 147.7 ms | 174.2 ms | 214.3 ms | 0 | 149.4 ms | 96 MB |
| text | pyox-agents | 10 | 15 | 7.2 ms | 21.8 ms | 668.6 ms | 736.9 ms | 768.9 ms | 0 | 106.0 ms | 60 MB |
| text | pydantic-ai | 10 | 4 | 58.8 ms | 95.0 ms | 2561.5 ms | 2607.9 ms | 2629.6 ms | 0 | 250.1 ms | 111 MB |
| text | openai-agents | 10 | 8 | 895.4 ms | 1271.4 ms | 1189.5 ms | 1338.8 ms | 1408.8 ms | 0 | 117.6 ms | 143 MB |
| text | agent-framework | 10 | 5 | 1250.6 ms | 2085.2 ms | 1835.3 ms | 2269.5 ms | 2621.0 ms | 0 | 196.2 ms | 167 MB |
| text | langgraph | 10 | 6 | 718.3 ms | 1449.5 ms | 1504.1 ms | 1810.9 ms | 2154.4 ms | 0 | 152.3 ms | 128 MB |
| text | langchain | 10 | 7 | 1201.5 ms | 1670.2 ms | 1515.2 ms | 1670.9 ms | 2320.6 ms | 0 | 152.0 ms | 133 MB |
| text | llama-index | 10 | 5 | 1175.6 ms | 2000.8 ms | 1946.7 ms | 2226.0 ms | 2478.6 ms | 0 | 199.6 ms | 199 MB |
| text | agno | 10 | 6 | 1307.5 ms | 1628.4 ms | 1598.2 ms | 1761.0 ms | 1844.3 ms | 0 | 158.9 ms | 98 MB |
| text | pyox-agents | 50 | 16 | 28.3 ms | 106.5 ms | 3063.5 ms | 3385.3 ms | 4048.2 ms | 0 | 95.3 ms | 72 MB |
| text | pydantic-ai | 50 | 3 | 379.8 ms | 623.0 ms | 14933.7 ms | 15054.2 ms | 15153.7 ms | 0 | 296.1 ms | 122 MB |
| text | openai-agents | 50 | 8 | 3997.8 ms | 6098.6 ms | 6124.4 ms | 6262.5 ms | 6282.8 ms | 0 | 120.2 ms | 156 MB |
| text | agent-framework | 50 | 5 | 5445.5 ms | 28882.5 ms | 6127.9 ms | 21394.7 ms | 29065.1 ms | 0 | 200.1 ms | 163 MB |
| text | langgraph | 50 | 6 | 3191.9 ms | 9082.7 ms | 8155.4 ms | 12117.2 ms | 14134.3 ms | 0 | 165.1 ms | 135 MB |
| text | langchain | 50 | 6 | 4355.5 ms | 8176.5 ms | 7967.1 ms | 9875.0 ms | 14255.5 ms | 0 | 161.1 ms | 148 MB |
| text | llama-index | 50 | 5 | 5933.5 ms | 11558.4 ms | 9653.7 ms | 12257.6 ms | 15493.1 ms | 0 | 198.4 ms | 208 MB |
| text | agno | 50 | 6 | 4654.4 ms | 26971.6 ms | 5105.1 ms | 15647.7 ms | 27241.6 ms | 0 | 157.7 ms | 101 MB |
| text | pyox-agents | 100 | 15 | 61.1 ms | 301.4 ms | 6048.7 ms | 8380.2 ms | 10654.7 ms | 0 | 101.7 ms | 80 MB |
| text | pydantic-ai | 100 | 3 | 582.8 ms | 1414.0 ms | 28790.4 ms | 28971.0 ms | 28977.3 ms | 0 | 287.5 ms | 135 MB |
| text | openai-agents | 100 | 8 | 6382.1 ms | 12006.2 ms | 11494.7 ms | 13547.4 ms | 20010.5 ms | 0 | 119.9 ms | 167 MB |
| text | agent-framework | 100 | 5 | 6250.6 ms | 38125.5 ms | 6996.0 ms | 34627.5 ms | 38297.5 ms | 2 | 197.3 ms | 175 MB |
| text | langgraph | 100 | 6 | 5691.6 ms | 25818.5 ms | 14080.5 ms | 25874.8 ms | 25883.6 ms | 0 | 161.1 ms | 143 MB |
| text | langchain | 100 | 6 | 6077.2 ms | 27996.7 ms | 12367.2 ms | 28032.9 ms | 28042.3 ms | 0 | 158.2 ms | 144 MB |
| text | llama-index | 100 | 5 | 13358.6 ms | 30031.4 ms | 18369.5 ms | 26379.5 ms | 34406.9 ms | 0 | 195.3 ms | 279 MB |
| text | agno | 100 | 7 | 5137.8 ms | 34243.3 ms | 5892.6 ms | 31227.0 ms | 34386.4 ms | 0 | 148.9 ms | 102 MB |
| tool | pyox-agents | 1 | 142 | 3.3 ms | 6.5 ms | 6.6 ms | 8.3 ms | 11.1 ms | 0 | 6.2 ms | 56 MB |
| tool | pydantic-ai | 1 | 28 | 21.7 ms | 33.9 ms | 34.1 ms | 43.3 ms | 52.3 ms | 0 | 33.3 ms | 107 MB |
| tool | openai-agents | 1 | 43 | 20.3 ms | 35.3 ms | 22.2 ms | 27.4 ms | 38.9 ms | 0 | 21.2 ms | 136 MB |
| tool | agent-framework | 1 | 41 | 13.9 ms | 24.5 ms | 22.6 ms | 28.9 ms | 39.7 ms | 0 | 22.1 ms | 94 MB |
| tool | langgraph | 1 | 36 | 24.3 ms | 40.1 ms | 26.0 ms | 34.0 ms | 42.3 ms | 0 | 25.1 ms | 121 MB |
| tool | langchain | 1 | 36 | 24.7 ms | 42.1 ms | 26.0 ms | 34.1 ms | 44.1 ms | 0 | 25.2 ms | 122 MB |
| tool | llama-index | 1 | 25 | 34.1 ms | 59.9 ms | 37.7 ms | 50.3 ms | 63.2 ms | 0 | 37.9 ms | 188 MB |
| tool | agno | 1 | 44 | 14.2 ms | 23.0 ms | 21.2 ms | 28.6 ms | 33.7 ms | 0 | 20.8 ms | 97 MB |
| tool | pyox-agents | 10 | 222 | 10.1 ms | 18.2 ms | 44.3 ms | 53.1 ms | 61.0 ms | 0 | 6.5 ms | 57 MB |
| tool | pydantic-ai | 10 | 30 | 183.4 ms | 319.9 ms | 331.6 ms | 389.8 ms | 468.8 ms | 0 | 33.1 ms | 112 MB |
| tool | openai-agents | 10 | 45 | 175.7 ms | 407.8 ms | 210.7 ms | 274.0 ms | 439.7 ms | 0 | 22.1 ms | 140 MB |
| tool | agent-framework | 10 | 44 | 208.0 ms | 454.0 ms | 216.7 ms | 262.9 ms | 462.2 ms | 0 | 22.6 ms | 111 MB |
| tool | langgraph | 10 | 41 | 166.2 ms | 253.5 ms | 236.2 ms | 292.8 ms | 340.0 ms | 0 | 24.4 ms | 123 MB |
| tool | langchain | 10 | 46 | 183.6 ms | 326.9 ms | 208.1 ms | 250.7 ms | 364.6 ms | 0 | 21.4 ms | 124 MB |
| tool | llama-index | 10 | 25 | 314.1 ms | 472.1 ms | 396.3 ms | 451.2 ms | 549.8 ms | 0 | 40.4 ms | 191 MB |
| tool | agno | 10 | 53 | 175.0 ms | 281.3 ms | 182.0 ms | 219.4 ms | 287.7 ms | 0 | 18.5 ms | 98 MB |
| tool | pyox-agents | 50 | 246 | 68.5 ms | 141.7 ms | 198.4 ms | 251.4 ms | 316.0 ms | 0 | 5.8 ms | 61 MB |
| tool | pydantic-ai | 50 | 27 | 1020.5 ms | 1364.3 ms | 1819.2 ms | 1955.4 ms | 2195.9 ms | 0 | 36.6 ms | 126 MB |
| tool | openai-agents | 50 | 47 | 794.8 ms | 1120.8 ms | 1049.8 ms | 1147.5 ms | 1365.7 ms | 0 | 21.3 ms | 150 MB |
| tool | agent-framework | 50 | 43 | 1068.4 ms | 2214.5 ms | 1077.4 ms | 1528.9 ms | 2235.6 ms | 0 | 23.1 ms | 146 MB |
| tool | langgraph | 50 | 39 | 866.9 ms | 1464.1 ms | 1252.9 ms | 1332.5 ms | 1786.7 ms | 0 | 25.1 ms | 127 MB |
| tool | langchain | 50 | 41 | 915.6 ms | 1291.1 ms | 1227.1 ms | 1297.1 ms | 1635.3 ms | 0 | 24.4 ms | 129 MB |
| tool | llama-index | 50 | 22 | 1732.7 ms | 2319.8 ms | 2213.2 ms | 2349.3 ms | 2814.7 ms | 0 | 44.7 ms | 196 MB |
| tool | agno | 50 | 49 | 992.9 ms | 1945.2 ms | 1000.8 ms | 1129.3 ms | 1953.0 ms | 0 | 20.2 ms | 103 MB |
| tool | pyox-agents | 100 | 220 | 137.6 ms | 386.6 ms | 403.0 ms | 735.9 ms | 1087.4 ms | 0 | 6.4 ms | 65 MB |
| tool | pydantic-ai | 100 | 22 | 2311.6 ms | 3369.9 ms | 4572.4 ms | 4920.8 ms | 5461.0 ms | 0 | 45.2 ms | 134 MB |
| tool | openai-agents | 100 | 40 | 1884.1 ms | 2909.9 ms | 2444.0 ms | 2861.8 ms | 3422.3 ms | 0 | 24.9 ms | 159 MB |
| tool | agent-framework | 100 | 37 | 2481.4 ms | 7000.8 ms | 2493.9 ms | 3055.5 ms | 7009.3 ms | 0 | 26.9 ms | 148 MB |
| tool | langgraph | 100 | 34 | 2167.5 ms | 3424.7 ms | 2759.0 ms | 3438.7 ms | 4181.7 ms | 0 | 29.2 ms | 136 MB |
| tool | langchain | 100 | 35 | 2058.5 ms | 4060.5 ms | 2813.9 ms | 3067.2 ms | 4856.3 ms | 0 | 28.1 ms | 132 MB |
| tool | llama-index | 100 | 23 | 3562.1 ms | 4514.7 ms | 4303.3 ms | 4686.7 ms | 5339.4 ms | 0 | 42.9 ms | 209 MB |
| tool | agno | 100 | 51 | 1902.1 ms | 2584.4 ms | 1908.8 ms | 2130.3 ms | 2591.8 ms | 0 | 19.5 ms | 109 MB |

</details>
<!-- server:end -->

- **Req/s**: complete answers per second.
- **First byte**: until the first piece of the answer reaches the client. The server waits for the first text
  before starting the response, so this is "time to first text" as a user sees it.
- **Latency**: until the whole answer has arrived.
- **CPU per request** (with 100 users) and **Peak memory**: measured inside the server process.
- **Failed**: requests without the full answer, over all the user counts.

What the numbers say:

- One uvicorn process is CPU-bound with every library: throughput stops growing at about 10 users, and from
  there latency grows with the number of users (requests wait their turn). pyox-agents reaches that ceiling
  later: 2 to 5 times more requests per second on long streamed answers (15 vs 3 to 8) and 4 to 10 times more
  with a tool call (about 220 vs 22 to 51).
- Time to first byte under load is where users feel it: with 100 users, the median wait before the answer starts
  is 61 ms with pyox-agents and 0.6 to 13 s with the others (streamed text), and 138 ms vs 1.9 to 3.6 s (tool call).
- With 100 users, the server process uses 65 to 80 MB with pyox-agents and 100 to 280 MB with the others.
- Two Agent Framework requests failed with 100 users: the `openai` client gave up connecting to the model after
  its 5 s connect timeout, because the overloaded process could not get to it in time.
- The server CPU per request is close to the library numbers above, so the agent library, not FastAPI or uvicorn,
  is most of the cost. To serve more, run more uvicorn workers or use a bigger `text_chunks`.

<!-- --8<-- [end:results] -->

## What is measured

Each scenario is one agent run with streaming on, going through every event (as an app showing tokens live):

| Scenario         | What the fake model does                                                        | Requests |
|------------------|---------------------------------------------------------------------------------|---------:|
| `text`           | Streams a long answer (5 paragraphs, ~1,800 characters in ~450 chunks)          | 1        |
| `tool`           | Asks for one tool call, then answers after getting the result                  | 2        |
| `parallel_tools` | Asks for five tool calls at once, then answers after getting the results      | 2        |

Every library gets the same system prompt and the same plain Python tool ([`libraries.py`](libraries.py)), and
each run checks that the final answer is the expected one, so a library can't look fast by skipping work.

## How each library is set up

We use each library's usual agent API with its default settings. We change only what is needed to talk to the
fake server (the Chat Completions API) and to stop data leaving the machine:

| Library         | Agent                                                    | Streamed text comes from                          | Changed from the defaults                          |
|-----------------|----------------------------------------------------------|---------------------------------------------------|----------------------------------------------------|
| pyox-agents     | `Agent` with a `Client`                                  | `TextDelta` events of `agent.execute()`           | nothing                                            |
| pydantic-ai     | `Agent` with `OpenAIChatModel`                           | text part events of `agent.run_stream_events()`   | nothing                                            |
| openai-agents   | `Agent` with `OpenAIChatCompletionsModel`                | `ResponseTextDeltaEvent`s of `Runner.run_streamed()` | Chat Completions model (the default is the Responses API); tracing off (it would upload traces to OpenAI) |
| agent-framework | `Agent` with `OpenAIChatCompletionClient`                | updates of `agent.run(stream=True)`               | Chat Completions client (`OpenAIChatClient` is the Responses API) |
| langgraph       | `StateGraph`: a model node and a `ToolNode` in a loop    | `AIMessageChunk`s of `astream(stream_mode="messages")` | nothing                                       |
| langchain       | `create_agent` with `ChatOpenAI`                         | the same as LangGraph (it builds a LangGraph graph) | nothing                                          |
| llama-index     | `FunctionAgent` with `OpenAI`                            | `AgentStream` events of `agent.run()`             | nothing                                            |
| agno            | `Agent` with `OpenAIChat`                                | `RunContent` events of `agent.arun(stream=True)`  | `telemetry=False` (it would send a request to Agno after every run) |

LangGraph's ready-made `create_react_agent` is deprecated in favour of LangChain's `create_agent`, so the
`langgraph` row is the graph LangGraph's docs build by hand, and `langchain` is the ready-made one.

We checked that the comparison is fair:

- Every library sends exactly the requests in the table above (counted on the fake server), and streams the text
  token by token (~450 pieces in `text`).
- Every library except pyox-agents runs the plain (sync) tool in a thread pool; pyox-agents calls it directly.
  The five calls of `parallel_tools` run at the same time in all of them, except LlamaIndex, which runs at most
  four at once by default.
- One agent is created once and shared by all runs (and all requests of the FastAPI app). Nothing is kept between
  runs: each run sends only the system prompt and the user message.
- Each library and scenario runs in its **own process**, so memory and caches never mix, and each measurement
  starts with 10 warm-up runs (connections opened, lazy imports done).
- Same Python, same machine and the same fake model for all. All the libraries except pyox-agents use the official
  `openai` client. pydantic-ai needs `openai` 3, so [`pyproject.toml`](pyproject.toml) overrides
  llama-index-llms-openai's `openai<3` requirement; LlamaIndex works with it.

## Tools

- **Fake model: [aimock](https://github.com/CopilotKit/aimock)** (`@copilotkit/aimock`, run with `npx`). A real HTTP
  server that speaks the OpenAI Chat Completions API, with streaming and tool calls. The answers are in
  [`fixtures.json`](fixtures.json). Matching is stateless (`hasToolResult`: has the conversation got a tool
  result?), so many runs can share the server at the same time. It costs about 0.4 ms per request, the same for
  every library.
- **Startup: [hyperfine](https://github.com/sharkdp/hyperfine)** (2.0 or newer). It runs a command many times and
  reports its mean time and memory, which is exactly the "start a new process" case.
- **Library overhead: a small asyncio script** ([`measure.py`](measure.py)). The libraries are *clients* used inside
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
  and then we would also be measuring that web server. The FastAPI benchmark does exactly that on purpose.
- **[llmposter](https://github.com/SkillDoAI/llmposter)** (a Rust mock server) was the first choice, but in
  version 0.5.0 every request on a reused connection waits ~40 ms (it doesn't disable Nagle's algorithm), which
  hides the libraries' own cost.
- **[ai-mock](https://pypi.org/project/ai-mock/)** streams one character per chunk, and as a Python server it
  could become the bottleneck under load.

## Measure one case by hand

```bash
npx --yes --package=@copilotkit/aimock@1.46.0 llmock -f fixtures.json -p 4010 -c 4 &
uv run measure.py langchain tool --runs 500

AGENT=langchain uv run uvicorn server:app --port 8000 &
k6 run -e URL=http://127.0.0.1:8000/chat -e PROMPT="Tell me about agents." -e EXPECTED=Agents \
  -e VUS=10 -e SUMMARY_FILE=summary.json load.js
```

## Files

| File                               | What it does                                                    |
|------------------------------------|-----------------------------------------------------------------|
| [`libraries.py`](libraries.py)     | The same agent written with each library                        |
| [`fixtures.json`](fixtures.json)   | The answers of the fake model                                   |
| [`measure.py`](measure.py)         | Measures one agent on one scenario and prints JSON              |
| [`run_all.py`](run_all.py)         | Runs the library benchmarks and writes the results              |
| [`server.py`](server.py)           | FastAPI app streaming an agent's answer (`POST /chat`)          |
| [`load.js`](load.js)               | k6 script: users sending requests to the app in a loop          |
| [`load_test.py`](load_test.py)     | Runs the FastAPI benchmark and writes the results               |

To add a library, add a `make_..._agent` function to `libraries.py` and an entry in `LIBRARIES` and `PACKAGES`.
Both benchmarks pick it up.
