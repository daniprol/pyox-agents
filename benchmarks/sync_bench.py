"""The sync API (`agent.run_sync`) against the async one, and against `asyncio.run` around it.

    uv run sync_bench.py                  # writes the table in README.md (~1 min)
    uv run sync_bench.py --runs 50        # quick, README unchanged

The ways, each running the `tool` scenario of `run_all.py` (one tool call, then the answer) to the end:
- `await agent.run(...)`: the async API, inside one event loop.
- `agent.run_sync(...)`: the sync API, no event loop.
- `asyncio.run(agent.run(...))`: what sync code would do without a sync API (a new loop per run).

Each is measured with 1 run at a time and 8 at a time: 8 tasks on one loop for the async API, 8 threads
(like a Gunicorn `gthread` worker) for the other two.
"""

import argparse
import asyncio
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from libraries import get_weather
from measure import SCENARIOS
from pyox_agents import Agent, Client
from run_all import BASE_URL, cpu_name, markdown_table, start_mock_server, stop_mock_server, write_into_readme

PROMPT = SCENARIOS["tool"][0]
CONCURRENCY = (1, 8)


def timed(run: Callable[[], object], latencies: list[float]) -> None:
    started = time.perf_counter()
    run()
    latencies.append(time.perf_counter() - started)


def measure_async(agent: Agent, runs: int, concurrency: int) -> tuple[list[float], float, float]:
    latencies: list[float] = []

    async def timed_run() -> None:
        started = time.perf_counter()
        await agent.run(PROMPT)
        latencies.append(time.perf_counter() - started)

    async def all_runs() -> tuple[float, float]:
        for _ in range(5):  # warm-up
            await agent.run(PROMPT)
        cpu_started, wall_started = time.process_time(), time.perf_counter()
        for _ in range(runs // concurrency):
            await asyncio.gather(*(timed_run() for _ in range(concurrency)))
        return time.process_time() - cpu_started, time.perf_counter() - wall_started

    cpu, wall = asyncio.run(all_runs())
    return latencies, cpu, wall


def measure_in_threads(run: Callable[[], object], runs: int, concurrency: int) -> tuple[list[float], float, float]:
    latencies: list[float] = []
    for _ in range(5):  # warm-up
        run()
    cpu_started, wall_started = time.process_time(), time.perf_counter()
    with ThreadPoolExecutor(concurrency) as threads:
        for _ in range(runs):
            threads.submit(timed, run, latencies)
    return latencies, time.process_time() - cpu_started, time.perf_counter() - wall_started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1000, help="runs per measurement (default 1000)")
    args = parser.parse_args()

    agent = Agent(client=Client(api_key="benchmark", base_url=BASE_URL), model="gpt-5-mini", tools=[get_weather])
    ways: dict[str, Callable[[int], tuple[list[float], float, float]]] = {
        "`await agent.run(...)`": lambda concurrency: measure_async(agent, args.runs, concurrency),
        "`agent.run_sync(...)`": lambda concurrency: measure_in_threads(
            lambda: agent.run_sync(PROMPT), args.runs, concurrency
        ),
        "`asyncio.run(agent.run(...))`": lambda concurrency: measure_in_threads(
            lambda: asyncio.run(agent.run(PROMPT)), args.runs, concurrency
        ),
    }

    server = start_mock_server()
    rows = []
    try:
        for concurrency in CONCURRENCY:
            for name, measure in ways.items():
                print(f"Measuring {name}, {concurrency} at a time...", flush=True)
                latencies, cpu, wall = measure(concurrency)
                rows.append([
                    name,
                    str(concurrency),
                    f"{statistics.median(latencies) * 1000:.2f} ms",
                    f"{cpu / len(latencies) * 1000:.2f} ms",
                    f"{len(latencies) / wall:.0f} runs/s",
                ])  # fmt: skip
    finally:
        stop_mock_server(server)

    headers = ["Way", "Runs at a time", "Latency p50", "CPU per run", "Throughput"]
    markdown = "\n".join(
        [f"{cpu_name()}. `tool` scenario, {args.runs} runs per row.", "", *markdown_table(headers, rows)]
    )
    print(markdown)
    if args.runs == parser.get_default("runs"):
        write_into_readme("sync", markdown)


if __name__ == "__main__":
    main()
