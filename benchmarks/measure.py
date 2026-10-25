"""Measure one library on one scenario, in this process, and print the results as JSON.

    python measure.py pyox-agents tool --base-url http://127.0.0.1:4010/v1

Two phases, each with its own numbers:

1. Sequential: one run after another. Gives the latency of a run and the CPU time it costs.
2. Concurrent: many runs at the same time, like a server with many users.
   Gives the throughput (runs per second) and the peak memory of the process.

`run_all.py` starts a new process for each library and scenario, so memory
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

from agents import LIBRARIES, RunAgent

os.environ["PYDANTIC_AI_NO_BANNER"] = "1"  # pydantic-ai prints a banner on its first run

# The prompt of each scenario and the final answer the fake server gives (see fixtures.json).
TOOL_ANSWER = "It is sunny everywhere you asked about, between 18 and 24 degrees."
SCENARIOS = {
    "text": ("Tell me about agents.", "Agents are programs"),  # one long streamed answer
    "tool": ("What's the weather in one city?", TOOL_ANSWER),  # one tool call, then the answer
    "parallel_tools": ("What's the weather in five cities?", TOOL_ANSWER),  # five tool calls at once
}


def peak_memory_mb() -> float:
    """Highest memory (RSS) this process has used so far."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 / 1024 if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux


async def run_checked(run: RunAgent, prompt: str, expected_start: str) -> None:
    """One run; fails loudly if the agent did not reach the expected final answer."""
    answer = await run(prompt)
    if not answer.startswith(expected_start):
        raise RuntimeError(f"Unexpected answer: {answer[:80]!r}")


async def measure(library: str, scenario: str, base_url: str, runs: int, concurrency: int) -> dict:
    prompt, expected = SCENARIOS[scenario]
    run = LIBRARIES[library](base_url)

    for _ in range(10):  # warm up connections and lazy imports
        await run_checked(run, prompt, expected)

    # 1. Sequential
    latencies_ms = []
    cpu_start = time.process_time()
    for _ in range(runs):
        start = time.perf_counter()
        await run_checked(run, prompt, expected)
        latencies_ms.append((time.perf_counter() - start) * 1000)
    cpu_ms_per_run = (time.process_time() - cpu_start) * 1000 / runs

    # 2. Concurrent: `concurrency` runs at once, `runs` in total
    limit = asyncio.Semaphore(concurrency)

    async def limited_run() -> None:
        async with limit:
            await run_checked(run, prompt, expected)

    start = time.perf_counter()
    await asyncio.gather(*(limited_run() for _ in range(runs)))
    concurrent_seconds = time.perf_counter() - start

    return {
        "library": library,
        "scenario": scenario,
        "latency_p50_ms": statistics.median(latencies_ms),
        "latency_p95_ms": statistics.quantiles(latencies_ms, n=20)[-1],
        "cpu_ms_per_run": cpu_ms_per_run,
        "runs_per_second": runs / concurrent_seconds,
        "memory_peak_mb": peak_memory_mb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", choices=LIBRARIES)
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument("--base-url", default="http://127.0.0.1:4010/v1")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()
    result = asyncio.run(measure(args.library, args.scenario, args.base_url, args.runs, args.concurrency))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
