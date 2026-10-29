# Benchmarks

The same agent (same prompt, tool and streamed answers) in pyox-agents and seven other Python agent frameworks:
[pydantic-ai](https://ai.pydantic.dev), [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/),
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework),
[LangGraph](https://langchain-ai.github.io/langgraph/), [LangChain](https://docs.langchain.com/oss/python/langchain/overview),
[LlamaIndex](https://developers.llamaindex.ai/python/framework/) and [Agno](https://docs.agno.com). A fake OpenAI
server answers instantly, so the numbers are each library's own overhead.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/), [Node.js](https://nodejs.org) 18+,
a [Rust toolchain](https://rustup.rs), [hyperfine](https://github.com/sharkdp/hyperfine#installation) 2.0+ and
[k6](https://grafana.com/docs/k6/latest/set-up/install-k6/) 1.x (Linux or macOS).

```bash
git clone https://github.com/daniprol/pyox-agents
cd pyox-agents/benchmarks
uv sync                                     # installs every library and builds pyox-agents
uv run run_all.py --runs 100                # library benchmark, quick (~10 min; full run ~1 h)
uv run run_all.py --only langgraph agno     # only some libraries, compared with pyox-agents
uv run load_test.py --duration 5s           # FastAPI benchmark, quick (~10 min; full run ~1 h)
```

Full runs rewrite the tables below; `--only` just prints them.

<!-- --8<-- [start:results] -->
## Called directly from Python

<!-- results:start -->
Python 3.13.17, Linux x86_64 (Intel(R) Xeon(R) Processor @ 2.80GHz). 500 runs per measurement, 100 at a time for throughput. Versions: pyox-agents 0.1.0, pydantic-ai 2.61.0, openai-agents 0.20.0, agent-framework 1.23.0, langgraph 1.2.18, langchain 1.4.7, llama-index 0.14.25, agno 3.1.12.

Latency p50 of one run, and in parentheses how many times the pyox-agents time it is. Fastest first.

| Library | Startup | `text` | `tool` | `parallel_tools` | Throughput, `tool` | Peak memory, `tool` |
|---|--:|--:|--:|--:|--:|--:|
| pyox-agents | 132 ms | 105.2 ms (1.0x) | 8.3 ms (1.0x) | 10.9 ms (1.0x) | 230 runs/s | 35 MB |
| agno | 1614 ms | 121.2 ms (1.2x) | 17.6 ms (2.1x) | 28.1 ms (2.6x) | 57 runs/s | 94 MB |
| agent-framework | 1462 ms | 147.7 ms (1.4x) | 21.7 ms (2.6x) | 32.6 ms (3.0x) | 40 runs/s | 129 MB |
| openai-agents | 3176 ms | 111.8 ms (1.1x) | 24.7 ms (3.0x) | 35.4 ms (3.3x) | 41 runs/s | 143 MB |
| langchain | 2760 ms | 151.1 ms (1.4x) | 27.2 ms (3.3x) | 44.1 ms (4.0x) | 40 runs/s | 125 MB |
| langgraph | 2752 ms | 158.6 ms (1.5x) | 27.6 ms (3.3x) | 46.2 ms (4.2x) | 37 runs/s | 123 MB |
| pydantic-ai | 2158 ms | 213.1 ms (2.0x) | 35.4 ms (4.2x) | 48.4 ms (4.4x) | 25 runs/s | 119 MB |
| llama-index | 3243 ms | 181.5 ms (1.7x) | 41.0 ms (4.9x) | 65.8 ms (6.0x) | 23 runs/s | 196 MB |

<details><summary>All the numbers</summary>

`text`: one long streamed answer

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

`tool`: one tool call, then the answer

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

`parallel_tools`: five tool calls at once, then the answer

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

Startup

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

</details>
<!-- results:end -->

Each scenario is one streamed agent run: `text` streams a long answer (~450 chunks), `tool` makes one tool call
and `parallel_tools` five at once, then answer. Startup is importing the library and creating an agent.
Throughput runs 100 at once in one process.

- **Tools:** pyox-agents is 2 to 6 times faster than every other library, serves 4 to 10 times more runs per
  second and uses a third to a fifth of the memory.
- **Long streamed text:** only 1.1 to 2 times faster with the default `text_chunks="delta"`, because each
  `TextDelta` event costs about 0.25 ms. With `text_chunks=10` it takes 15 ms, 7 to 14 times less than the others.
- **Startup:** 0.13 s vs 1.5 to 3.2 s.
- "First text" for `text` is close to the whole run for the OpenAI Agents SDK, LangChain, LangGraph and LlamaIndex.
  This is most likely an artifact of the instant fake model: they pass pieces through a queue between asyncio tasks.

## Streaming modes (`text_chunks`)

pyox-agents can group streamed text into fewer `TextDelta` events. Cost of each mode on the `text` scenario:

<!-- text-chunks:start -->
| Mode | Latency p50 | Latency p95 | First text | CPU per run | Throughput | Failed | Peak memory |
|---|--:|--:|--:|--:|--:|--:|--:|
| text_chunks='paragraph' | 6.7 ms | 8.6 ms | 4.0 ms | 6.4 ms | 432 runs/s | 0 | 48 MB |
| text_chunks='message' | 7.6 ms | 9.8 ms | 6.8 ms | 6.1 ms | 409 runs/s | 0 | 46 MB |
| text_chunks=10 | 14.7 ms | 21.3 ms | 3.2 ms | 15.5 ms | 112 runs/s | 0 | 57 MB |
| text_chunks='delta' | 111.1 ms | 142.1 ms | 3.3 ms | 125.0 ms | 12 runs/s | 0 | 49 MB |
<!-- text-chunks:end -->

The cost is per event, not per character. With a real model, the wait for the first `TextDelta` grows with the
group size (the whole answer with `"message"`), so `10` is the safe choice for live UIs.

## Behind a FastAPI streaming endpoint

Each library serves `POST /chat` from one uvicorn process
([`server.py`](https://github.com/daniprol/pyox-agents/blob/main/benchmarks/server.py)), and [k6](https://k6.io)
users each send requests in a loop, reading the whole streamed answer
([`load.js`](https://github.com/daniprol/pyox-agents/blob/main/benchmarks/load.js)). "First byte" is when the first
text arrives; "Failed" counts requests without the full answer.

<!-- server:start -->
Python 3.13.17, one uvicorn process, Intel(R) Xeon(R) Processor @ 2.80GHz. Each measurement: 20s of load.

### `text`: one long streamed answer

| Library | Req/s, 1 user | Req/s, 10 users | Req/s, 50 users | Req/s, 100 users | First byte p50, 100 users | Latency avg / p50 / p90 / p99, 100 users | CPU per request | Peak memory | Failed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pyox-agents | 11 | 16 | 16 | 17 | 49 ms | 5.6 / 5.5 / 7.4 / 9.3 s | 90.2 ms | 83 MB | 0 |
| openai-agents | 8 | 8 | 8 | 8 | 6442 ms | 11.1 / 10.3 / 18.2 / 20.9 s | 119.2 ms | 153 MB | 0 |
| agno | 7 | 7 | 7 | 7 | 5318 ms | 12.1 / 5.9 / 31.9 / 34.9 s | 151.5 ms | 105 MB | 0 |
| langchain | 6 | 7 | 7 | 6 | 6021 ms | 14.8 / 12.2 / 28.5 / 28.5 s | 154.2 ms | 147 MB | 0 |
| langgraph | 6 | 7 | 6 | 6 | 10360 ms | 15.8 / 13.5 / 26.5 / 26.8 s | 158.8 ms | 191 MB | 0 |
| llama-index | 5 | 5 | 5 | 5 | 9192 ms | 17.0 / 15.6 / 30.0 / 30.0 s | 191.2 ms | 216 MB | 0 |
| agent-framework | 5 | 5 | 5 | 5 | 10114 ms | 15.5 / 12.4 / 32.5 / 37.0 s | 199.9 ms | 214 MB | 0 |
| pydantic-ai | 4 | 4 | 4 | 3 | 922 ms | 28.7 / 28.8 / 29.0 / 29.0 s | 287.3 ms | 134 MB | 0 |

### `tool`: one tool call, then the answer

| Library | Req/s, 1 user | Req/s, 10 users | Req/s, 50 users | Req/s, 100 users | First byte p50, 100 users | Latency avg / p50 / p90 / p99, 100 users | CPU per request | Peak memory | Failed |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pyox-agents | 145 | 234 | 254 | 247 | 118 ms | 0.4 / 0.4 / 0.6 / 0.9 s | 5.8 ms | 65 MB | 0 |
| agno | 49 | 56 | 54 | 56 | 1729 ms | 1.7 / 1.7 / 1.9 / 6.2 s | 17.7 ms | 107 MB | 0 |
| openai-agents | 44 | 50 | 48 | 47 | 1627 ms | 2.1 / 2.1 / 2.4 / 2.7 s | 21.3 ms | 159 MB | 0 |
| agent-framework | 46 | 46 | 42 | 42 | 2152 ms | 2.3 / 2.2 / 2.7 / 8.6 s | 23.5 ms | 148 MB | 0 |
| langchain | 40 | 46 | 41 | 41 | 1770 ms | 2.4 / 2.4 / 2.6 / 4.1 s | 24.1 ms | 133 MB | 0 |
| langgraph | 36 | 42 | 42 | 39 | 1804 ms | 2.5 / 2.5 / 2.7 / 4.0 s | 25.2 ms | 133 MB | 0 |
| pydantic-ai | 30 | 33 | 30 | 28 | 2042 ms | 3.5 / 3.6 / 3.7 / 3.9 s | 34.9 ms | 134 MB | 0 |
| llama-index | 27 | 24 | 25 | 24 | 3262 ms | 4.1 / 4.1 / 4.4 / 6.8 s | 42.2 ms | 200 MB | 0 |

<details><summary>All the numbers</summary>

| Scenario | Library | Users | Requests/s | First byte p50 | p99 | Latency avg | p50 | p90 | p99 | Failed | Server CPU per request | Server peak memory |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| text | pyox-agents | 1 | 11 | 4.0 ms | 8.9 ms | 92.0 ms | 91.4 ms | 101.8 ms | 115.4 ms | 0 | 101.9 ms | 56 MB |
| text | pydantic-ai | 1 | 4 | 10.9 ms | 17.7 ms | 248.7 ms | 243.5 ms | 279.4 ms | 316.8 ms | 0 | 242.9 ms | 106 MB |
| text | openai-agents | 1 | 8 | 67.9 ms | 126.5 ms | 129.4 ms | 122.7 ms | 155.0 ms | 210.4 ms | 0 | 124.9 ms | 134 MB |
| text | agent-framework | 1 | 5 | 8.5 ms | 21.8 ms | 201.3 ms | 187.8 ms | 236.1 ms | 390.7 ms | 0 | 195.9 ms | 120 MB |
| text | langgraph | 1 | 6 | 137.4 ms | 287.3 ms | 157.9 ms | 149.1 ms | 185.2 ms | 300.1 ms | 0 | 152.8 ms | 122 MB |
| text | langchain | 1 | 6 | 134.8 ms | 286.1 ms | 155.2 ms | 145.9 ms | 183.4 ms | 295.7 ms | 0 | 150.4 ms | 123 MB |
| text | llama-index | 1 | 5 | 159.4 ms | 372.9 ms | 188.5 ms | 177.9 ms | 215.3 ms | 388.0 ms | 0 | 183.2 ms | 188 MB |
| text | agno | 1 | 7 | 7.8 ms | 12.0 ms | 149.5 ms | 144.3 ms | 167.3 ms | 220.2 ms | 0 | 144.9 ms | 96 MB |
| text | pyox-agents | 10 | 16 | 6.7 ms | 31.1 ms | 630.9 ms | 643.5 ms | 693.6 ms | 726.2 ms | 0 | 100.7 ms | 59 MB |
| text | pydantic-ai | 10 | 4 | 67.9 ms | 97.1 ms | 2314.1 ms | 2324.0 ms | 2432.0 ms | 2441.3 ms | 0 | 230.0 ms | 110 MB |
| text | openai-agents | 10 | 8 | 856.9 ms | 1257.5 ms | 1189.9 ms | 1144.5 ms | 1332.8 ms | 2121.0 ms | 0 | 117.7 ms | 141 MB |
| text | agent-framework | 10 | 5 | 1087.9 ms | 2096.5 ms | 1932.2 ms | 1945.6 ms | 2355.7 ms | 2410.1 ms | 0 | 196.9 ms | 167 MB |
| text | langgraph | 10 | 7 | 847.1 ms | 1554.1 ms | 1497.3 ms | 1486.6 ms | 1603.2 ms | 2553.4 ms | 0 | 148.0 ms | 129 MB |
| text | langchain | 10 | 7 | 1136.8 ms | 1626.5 ms | 1487.3 ms | 1448.9 ms | 1573.6 ms | 1818.1 ms | 0 | 147.3 ms | 132 MB |
| text | llama-index | 10 | 5 | 1192.2 ms | 2587.2 ms | 1925.1 ms | 1840.7 ms | 2349.4 ms | 2697.0 ms | 0 | 195.3 ms | 208 MB |
| text | agno | 10 | 7 | 774.3 ms | 1386.7 ms | 1449.5 ms | 1437.6 ms | 1561.7 ms | 1576.9 ms | 0 | 145.0 ms | 98 MB |
| text | pyox-agents | 50 | 16 | 27.4 ms | 133.2 ms | 2991.4 ms | 2970.7 ms | 3376.2 ms | 4048.2 ms | 0 | 94.1 ms | 71 MB |
| text | pydantic-ai | 50 | 4 | 364.1 ms | 633.8 ms | 13763.6 ms | 13707.1 ms | 14060.0 ms | 14147.9 ms | 0 | 273.1 ms | 122 MB |
| text | openai-agents | 50 | 8 | 3209.6 ms | 5702.4 ms | 5980.1 ms | 6015.0 ms | 6340.2 ms | 6466.9 ms | 0 | 118.8 ms | 146 MB |
| text | agent-framework | 50 | 5 | 5451.6 ms | 29112.8 ms | 8244.7 ms | 5917.5 ms | 21448.1 ms | 29600.5 ms | 0 | 193.5 ms | 168 MB |
| text | langgraph | 50 | 6 | 7150.2 ms | 8103.6 ms | 7756.4 ms | 7565.1 ms | 8215.8 ms | 12308.4 ms | 0 | 153.7 ms | 182 MB |
| text | langchain | 50 | 7 | 3314.9 ms | 10508.5 ms | 7357.6 ms | 7518.7 ms | 11313.6 ms | 13906.4 ms | 0 | 149.9 ms | 137 MB |
| text | llama-index | 50 | 5 | 5663.2 ms | 10096.0 ms | 9405.9 ms | 9435.8 ms | 11356.8 ms | 15557.6 ms | 0 | 197.4 ms | 208 MB |
| text | agno | 50 | 7 | 4554.5 ms | 9771.1 ms | 6720.6 ms | 6995.7 ms | 7902.2 ms | 14473.5 ms | 0 | 148.2 ms | 105 MB |
| text | pyox-agents | 100 | 17 | 49.0 ms | 308.8 ms | 5585.5 ms | 5495.0 ms | 7427.6 ms | 9287.9 ms | 0 | 90.2 ms | 83 MB |
| text | pydantic-ai | 100 | 3 | 922.3 ms | 1755.2 ms | 28749.9 ms | 28810.1 ms | 28987.7 ms | 28992.4 ms | 0 | 287.3 ms | 134 MB |
| text | openai-agents | 100 | 8 | 6442.5 ms | 17313.3 ms | 11065.4 ms | 10319.9 ms | 18228.4 ms | 20940.6 ms | 0 | 119.2 ms | 153 MB |
| text | agent-framework | 100 | 5 | 10114.4 ms | 36805.1 ms | 15504.1 ms | 12423.7 ms | 32546.1 ms | 36973.6 ms | 0 | 199.9 ms | 214 MB |
| text | langgraph | 100 | 6 | 10359.7 ms | 25366.8 ms | 15758.9 ms | 13529.0 ms | 26544.3 ms | 26754.9 ms | 0 | 158.8 ms | 191 MB |
| text | langchain | 100 | 6 | 6020.7 ms | 28418.8 ms | 14817.7 ms | 12175.4 ms | 28463.9 ms | 28473.8 ms | 0 | 154.2 ms | 147 MB |
| text | llama-index | 100 | 5 | 9192.4 ms | 29762.6 ms | 17032.4 ms | 15578.6 ms | 29953.6 ms | 29996.2 ms | 0 | 191.2 ms | 216 MB |
| text | agno | 100 | 7 | 5317.7 ms | 34796.8 ms | 12129.1 ms | 5927.4 ms | 31883.1 ms | 34937.2 ms | 0 | 151.5 ms | 101 MB |
| tool | pyox-agents | 1 | 145 | 3.2 ms | 6.0 ms | 6.7 ms | 6.4 ms | 8.1 ms | 11.2 ms | 0 | 6.2 ms | 55 MB |
| tool | pydantic-ai | 1 | 30 | 19.9 ms | 29.9 ms | 33.1 ms | 31.8 ms | 38.2 ms | 47.1 ms | 0 | 31.1 ms | 107 MB |
| tool | openai-agents | 1 | 44 | 19.7 ms | 31.3 ms | 22.7 ms | 21.6 ms | 28.9 ms | 34.0 ms | 0 | 20.8 ms | 135 MB |
| tool | agent-framework | 1 | 46 | 12.3 ms | 19.5 ms | 21.6 ms | 20.4 ms | 26.1 ms | 33.4 ms | 0 | 20.1 ms | 93 MB |
| tool | langgraph | 1 | 36 | 23.7 ms | 37.1 ms | 27.3 ms | 25.3 ms | 32.9 ms | 39.7 ms | 0 | 24.4 ms | 120 MB |
| tool | langchain | 1 | 40 | 22.7 ms | 33.5 ms | 25.0 ms | 23.9 ms | 30.8 ms | 35.4 ms | 0 | 23.2 ms | 122 MB |
| tool | llama-index | 1 | 27 | 32.0 ms | 49.3 ms | 37.4 ms | 35.4 ms | 44.8 ms | 54.0 ms | 0 | 35.5 ms | 186 MB |
| tool | agno | 1 | 49 | 12.3 ms | 21.2 ms | 20.2 ms | 18.9 ms | 25.7 ms | 30.9 ms | 0 | 18.8 ms | 97 MB |
| tool | pyox-agents | 10 | 234 | 9.5 ms | 16.9 ms | 42.6 ms | 42.0 ms | 51.0 ms | 59.5 ms | 0 | 6.2 ms | 56 MB |
| tool | pydantic-ai | 10 | 33 | 171.8 ms | 256.3 ms | 304.1 ms | 296.1 ms | 354.6 ms | 473.7 ms | 0 | 30.3 ms | 111 MB |
| tool | openai-agents | 10 | 50 | 160.5 ms | 264.7 ms | 199.2 ms | 191.7 ms | 231.4 ms | 297.8 ms | 0 | 19.9 ms | 140 MB |
| tool | agent-framework | 10 | 46 | 197.6 ms | 422.6 ms | 215.5 ms | 206.3 ms | 248.8 ms | 441.9 ms | 0 | 21.5 ms | 110 MB |
| tool | langgraph | 10 | 42 | 167.2 ms | 236.5 ms | 240.2 ms | 237.4 ms | 271.9 ms | 312.3 ms | 0 | 23.9 ms | 122 MB |
| tool | langchain | 10 | 46 | 190.5 ms | 325.6 ms | 218.8 ms | 212.6 ms | 254.8 ms | 343.9 ms | 0 | 21.6 ms | 124 MB |
| tool | llama-index | 10 | 24 | 333.6 ms | 499.1 ms | 418.1 ms | 414.1 ms | 473.6 ms | 584.4 ms | 0 | 42.2 ms | 189 MB |
| tool | agno | 10 | 56 | 166.4 ms | 253.7 ms | 176.9 ms | 173.2 ms | 201.0 ms | 260.1 ms | 0 | 17.6 ms | 98 MB |
| tool | pyox-agents | 50 | 254 | 67.5 ms | 140.2 ms | 196.2 ms | 192.9 ms | 242.6 ms | 306.0 ms | 0 | 5.7 ms | 61 MB |
| tool | pydantic-ai | 50 | 30 | 929.2 ms | 1160.7 ms | 1673.7 ms | 1673.6 ms | 1783.1 ms | 1890.5 ms | 0 | 33.4 ms | 127 MB |
| tool | openai-agents | 50 | 48 | 771.4 ms | 1013.5 ms | 1038.2 ms | 1024.8 ms | 1134.9 ms | 1260.5 ms | 0 | 20.9 ms | 148 MB |
| tool | agent-framework | 50 | 42 | 1105.6 ms | 2208.7 ms | 1163.0 ms | 1115.0 ms | 1571.5 ms | 2216.4 ms | 0 | 23.5 ms | 148 MB |
| tool | langgraph | 50 | 42 | 813.2 ms | 1216.7 ms | 1175.9 ms | 1170.2 ms | 1258.8 ms | 1525.1 ms | 0 | 23.6 ms | 126 MB |
| tool | langchain | 50 | 41 | 901.1 ms | 1276.0 ms | 1218.3 ms | 1211.0 ms | 1313.5 ms | 1537.1 ms | 0 | 24.4 ms | 127 MB |
| tool | llama-index | 50 | 25 | 1552.7 ms | 2251.9 ms | 1956.2 ms | 1990.4 ms | 2130.2 ms | 2684.0 ms | 0 | 40.0 ms | 194 MB |
| tool | agno | 50 | 54 | 900.9 ms | 1826.4 ms | 908.7 ms | 908.1 ms | 966.4 ms | 1835.0 ms | 0 | 18.2 ms | 102 MB |
| tool | pyox-agents | 100 | 247 | 118.0 ms | 339.5 ms | 403.2 ms | 384.9 ms | 603.2 ms | 898.1 ms | 0 | 5.8 ms | 65 MB |
| tool | pydantic-ai | 100 | 28 | 2042.2 ms | 2371.4 ms | 3504.6 ms | 3601.8 ms | 3656.3 ms | 3914.9 ms | 0 | 34.9 ms | 134 MB |
| tool | openai-agents | 100 | 47 | 1626.7 ms | 2305.2 ms | 2097.8 ms | 2108.6 ms | 2417.1 ms | 2696.6 ms | 0 | 21.3 ms | 159 MB |
| tool | agent-framework | 100 | 42 | 2152.0 ms | 8563.1 ms | 2277.1 ms | 2160.5 ms | 2723.8 ms | 8580.1 ms | 0 | 23.5 ms | 148 MB |
| tool | langgraph | 100 | 39 | 1804.2 ms | 3065.2 ms | 2496.9 ms | 2535.7 ms | 2671.4 ms | 3999.3 ms | 0 | 25.2 ms | 133 MB |
| tool | langchain | 100 | 41 | 1769.5 ms | 3465.5 ms | 2409.8 ms | 2393.6 ms | 2583.5 ms | 4149.1 ms | 0 | 24.1 ms | 133 MB |
| tool | llama-index | 100 | 24 | 3261.9 ms | 5886.3 ms | 4065.0 ms | 4121.8 ms | 4444.7 ms | 6754.5 ms | 0 | 42.2 ms | 200 MB |
| tool | agno | 100 | 56 | 1729.1 ms | 6242.5 ms | 1736.7 ms | 1735.9 ms | 1889.1 ms | 6249.5 ms | 0 | 17.7 ms | 107 MB |

</details>
<!-- server:end -->

- One process is CPU-bound with every library, so throughput stops growing at about 10 users. pyox-agents gets
  2 to 6 times more requests per second on streamed text and 4 to 10 times more with a tool call.
- With 100 users, pyox-agents starts every answer within about 50 ms and streams them side by side. The others
  make most answers wait 1 to 10 seconds before they start. Agno's p50 is as good as pyox-agents' on streamed text,
  but it runs a started answer to the end while the others wait, so its p90 and p99 are about 4 times worse.
  The average is the fairest single number.

<!-- --8<-- [end:results] -->

## How it's kept fair

| Library         | Agent                                         | Changed from the defaults                                        |
|-----------------|-----------------------------------------------|------------------------------------------------------------------|
| pyox-agents     | `Agent` with a `Client`                       |                                                                  |
| pydantic-ai     | `Agent` with `OpenAIChatModel`                |                                                                  |
| openai-agents   | `Agent` with `OpenAIChatCompletionsModel`     | Chat Completions instead of the Responses API; tracing off       |
| agent-framework | `Agent` with `OpenAIChatCompletionClient`     | Chat Completions instead of the Responses API                    |
| langgraph       | `StateGraph` with a model node and a `ToolNode` | (LangGraph's `create_react_agent` is deprecated)               |
| langchain       | `create_agent` with `ChatOpenAI`              |                                                                  |
| llama-index     | `FunctionAgent` with `OpenAI`                 |                                                                  |
| agno            | `Agent` with `OpenAIChat`                     | `telemetry=False`                                                |

- The fake server only speaks Chat Completions, and tracing/telemetry would send data out; nothing else changes.
- Every library sends the same requests (1 for `text`, 2 with tools), streams token by token, and must produce the
  expected final answer.
- One agent is shared by all runs, with no history between them. Every library but pyox-agents runs the sync tool
  in a thread pool; LlamaIndex runs at most 4 tool calls at once.
- Each library and scenario runs in its own process, after 10 warm-up runs.
- [`pyproject.toml`](pyproject.toml) overrides llama-index-llms-openai's `openai<3` (pydantic-ai needs `openai` 3).

Tools: [aimock](https://github.com/CopilotKit/aimock) is the fake model (answers in [`fixtures.json`](fixtures.json)),
[hyperfine](https://github.com/sharkdp/hyperfine) measures startup, [`measure.py`](measure.py) measures the library
in-process and [k6](https://k6.io) loads the FastAPI app. [llmposter](https://github.com/SkillDoAI/llmposter) 0.5.0
was rejected (~40 ms delay on reused connections).

## Files

| File                           | What it does                                                       |
|--------------------------------|--------------------------------------------------------------------|
| [`libraries.py`](libraries.py) | The same agent in each library (add one here, in `LIBRARIES` and `PACKAGES`) |
| [`measure.py`](measure.py)     | Measures one agent on one scenario: `uv run measure.py agno tool` (fake server running) |
| [`run_all.py`](run_all.py)     | Runs the library benchmarks and writes the results                 |
| [`server.py`](server.py)       | FastAPI app streaming an agent's answer                            |
| [`load.js`](load.js)           | k6 script: users sending requests in a loop                        |
| [`load_test.py`](load_test.py) | Runs the FastAPI benchmark and writes the results                  |
