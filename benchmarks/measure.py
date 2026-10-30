"""Measure one agent on one scenario, in this process, and print the results as JSON.

    python measure.py pyox-agents tool --base-url http://127.0.0.1:4010/v1
    python measure.py "text_chunks='paragraph'" text

Two phases, each with its own numbers:

1. Sequential: one run after another. Gives the latency of a run, the time until its first
   text and the CPU time it costs.
2. Concurrent: many runs at the same time, like a server with many users.
   Gives the throughput (runs per second), failed runs and the peak memory of the process.

`run_all.py` starts a new process for each agent and scenario, so memory
numbers never mix.
"""

import argparse
import asyncio
import json
import os
import resource
import statistics
import sys
import time

from libraries import AGENTS, StreamText

os.environ["PYDANTIC_AI_NO_BANNER"] = "1"  # pydantic-ai prints a banner on its first run

# The prompt of each scenario and the final answer the fake server gives (see fixtures.json).
TOOL_ANSWER = "It is sunny everywhere you asked about, between 18 and 24 degrees."
SCENARIOS = {
    "text": ("Tell me about agents.", "Agents are programs"),  # one long streamed answer
    "tool": ("What's the weather in one city?", TOOL_ANSWER),  # one tool call, then the answer
    "parallel_tools": ("What's the weather in five cities?", TOOL_ANSWER),  # five tool calls at once
}
# For the Responses API server of `websocket_bench.py`: tool calls one after another, one per turn.
TOOL_LOOP_SCENARIOS = {
    f"tool_loop_{calls}": (f"Check the weather in Vigo with {calls} tool calls, one at a time.", TOOL_ANSWER)
    for calls in (1, 5, 20)
}
ALL_SCENARIOS = SCENARIOS | TOOL_LOOP_SCENARIOS


def peak_memory_mb() -> float:
    """Highest memory (RSS) this process has used so far."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 / 1024 if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux


async def run_checked(stream: StreamText, prompt: str, expected_start: str) -> float:
    """One run, reading every piece of text. Returns the seconds until the first piece arrived.

    Fails loudly if the agent did not reach the expected final answer.
    """
    start = time.perf_counter()
    first_text_seconds = 0.0
    answer = ""
    async for text in stream(prompt):
        if not answer:
            first_text_seconds = time.perf_counter() - start
        answer += text
    if not answer.startswith(expected_start):
        raise RuntimeError(f"Unexpected answer: {answer[:80]!r}")
    return first_text_seconds


async def measure(agent: str, scenario: str, base_url: str, runs: int, concurrency: int, any_answer: bool) -> dict:
    prompt, expected = ALL_SCENARIOS[scenario]
    if any_answer:  # a real model words its answer its own way
        expected = ""
    stream = AGENTS[agent](base_url)

    for _ in range(10):  # warm up connections and lazy imports
        await run_checked(stream, prompt, expected)

    # 1. Sequential
    latencies_ms = []
    first_text_ms = []
    cpu_start = time.process_time()
    for _ in range(runs):
        start = time.perf_counter()
        first_text_seconds = await run_checked(stream, prompt, expected)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        first_text_ms.append(first_text_seconds * 1000)
    cpu_ms_per_run = (time.process_time() - cpu_start) * 1000 / runs

    # 2. Concurrent: `concurrency` runs at once, `runs` in total.
    # Under load a run may fail (for example a dropped connection): count it instead of stopping.
    limit = asyncio.Semaphore(concurrency)
    failed_runs = 0

    async def limited_run() -> None:
        nonlocal failed_runs
        async with limit:
            try:
                await run_checked(stream, prompt, expected)
            except Exception as error:
                failed_runs += 1
                print(f"Run failed: {error!r}", file=sys.stderr)

    start = time.perf_counter()
    await asyncio.gather(*(limited_run() for _ in range(runs)))
    concurrent_seconds = time.perf_counter() - start

    return {
        "agent": agent,
        "scenario": scenario,
        "first_text_p50_ms": statistics.median(first_text_ms),
        "latency_p50_ms": statistics.median(latencies_ms),
        "latency_p95_ms": statistics.quantiles(latencies_ms, n=20)[-1],
        "cpu_ms_per_run": cpu_ms_per_run,
        "runs_per_second": (runs - failed_runs) / concurrent_seconds,
        "failed_runs": failed_runs,
        "memory_peak_mb": peak_memory_mb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent", choices=AGENTS)
    parser.add_argument("scenario", choices=ALL_SCENARIOS)
    parser.add_argument("--base-url", default="http://127.0.0.1:4010/v1")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--any-answer", action="store_true", help="accept any final answer (for a real model)")
    args = parser.parse_args()
    result = asyncio.run(
        measure(args.agent, args.scenario, args.base_url, args.runs, args.concurrency, args.any_answer)
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
