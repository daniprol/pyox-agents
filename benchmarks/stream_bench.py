"""Three ways to serve a run as an AI SDK stream: does `run.batches()` or `run.ai_sdk()` pay off?

    uv run stream_bench.py                  # writes the tables in README.md (~2 min)
    uv run stream_bench.py --runs 50        # quick, README unchanged

The ways (each makes a `text-delta` chunk per streamed piece of text):
- `async for event in run`: one Python step per event, the chunks made in Python.
- `async for events in run.batches()`: one step per batch of waiting events, the chunks made in Python.
- `async for text in run.ai_sdk()`: one step per batch, the chunks made in Rust.

Each is measured with 1 run at a time (one user) and 50 at a time (a busy server), against two fake
OpenAI servers, each in its own process:
- instant: the `text` answer of `run_all.py` (~450 pieces), all at once. The cost of the Python side alone.
- paced: 100 pieces, 10 ms apart, like a fast real model (`paced_server.py`).
"""

import argparse
import asyncio
import json
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable

from pyox_agents import Agent, Client, RunFinished, TextDelta, ToolFinished, ToolStarted
from run_all import BASE_URL, HERE, cpu_name, markdown_table, start_mock_server, stop_mock_server, write_into_readme

PROMPT = "Tell me about agents."  # the `text` scenario: one long streamed answer
CONCURRENCY = (1, 50)
PACED_PORT = 4030


def start_paced_server() -> subprocess.Popen:
    server = subprocess.Popen([sys.executable, "paced_server.py", str(PACED_PORT)], cwd=HERE)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", PACED_PORT), timeout=1).close()
            return server
        except OSError:
            time.sleep(0.1)
    server.kill()
    raise RuntimeError("The paced server did not start")


def ai_sdk_chunk(event: object) -> str:
    """The AI SDK chunk for an event, made in Python (the text part only, enough to compare)."""
    if isinstance(event, TextDelta):
        data = {"type": "text-delta", "id": "text-1", "delta": event.text}
    elif isinstance(event, ToolStarted):
        data = {"type": "tool-input-available", "toolCallId": event.call_id, "toolName": event.name}
    elif isinstance(event, ToolFinished):
        data = {"type": "tool-output-available", "toolCallId": event.call_id, "output": event.result}
    elif isinstance(event, RunFinished):
        data = {"type": "finish"}
    else:
        return ""
    return f"data: {json.dumps(data)}\n\n"


async def events_in_python(agent: Agent) -> int:
    steps = 0
    async for event in agent.execute(PROMPT):
        ai_sdk_chunk(event)
        steps += 1
    return steps


async def batches_in_python(agent: Agent) -> int:
    steps = 0
    async for events in agent.execute(PROMPT).batches():
        "".join(ai_sdk_chunk(event) for event in events)
        steps += 1
    return steps


async def ai_sdk_in_rust(agent: Agent) -> int:
    steps = 0
    async for _text in agent.execute(PROMPT).ai_sdk():
        steps += 1
    return steps


WAYS: dict[str, Callable[[Agent], Awaitable[int]]] = {
    "`async for event in run` + Python": events_in_python,
    "`run.batches()` + Python": batches_in_python,
    "`run.ai_sdk()` (Rust)": ai_sdk_in_rust,
}


async def measure(
    stream: Callable[[Agent], Awaitable[int]], agent: Agent, runs: int, concurrency: int
) -> dict[str, float]:
    for _ in range(5):  # warm-up
        await stream(agent)
    latencies: list[float] = []
    steps: list[int] = []

    async def timed_run() -> None:
        started = time.perf_counter()
        steps.append(await stream(agent))
        latencies.append(time.perf_counter() - started)

    cpu_started, wall_started = time.process_time(), time.perf_counter()
    for _ in range(runs // concurrency or 1):
        await asyncio.gather(*(timed_run() for _ in range(concurrency)))
    cpu, wall = time.process_time() - cpu_started, time.perf_counter() - wall_started
    return {
        "latency_ms": statistics.median(latencies) * 1000,
        "cpu_ms": cpu / len(latencies) * 1000,
        "runs_per_s": len(latencies) / wall,
        "steps": statistics.mean(steps),
    }


async def measure_all(base_url: str, runs: int) -> list[str]:
    """A Markdown table: each way, at each concurrency."""
    agent = Agent(client=Client(api_key="benchmark", base_url=base_url), model="gpt-4o-mini")
    rows = []
    for concurrency in CONCURRENCY:
        for name, stream in WAYS.items():
            print(f"Measuring {name}, {concurrency} at a time...", flush=True)
            result = await measure(stream, agent, runs, concurrency)
            rows.append([
                name,
                str(concurrency),
                f"{result['latency_ms']:.1f} ms",
                f"{result['cpu_ms']:.1f} ms",
                f"{result['runs_per_s']:.0f} runs/s",
                f"{result['steps']:.0f}",
            ])  # fmt: skip
    headers = ["Way", "Runs at a time", "Latency p50", "CPU per run", "Throughput", "Python steps per run"]
    return markdown_table(headers, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=500, help="runs per measurement (default 500)")
    args = parser.parse_args()

    server = start_mock_server()
    try:
        instant = asyncio.run(measure_all(BASE_URL, args.runs))
    finally:
        stop_mock_server(server)
    server = start_paced_server()
    try:
        paced_runs = max(args.runs // 25, 10)  # each run takes 1 s here
        paced = asyncio.run(measure_all(f"http://127.0.0.1:{PACED_PORT}/v1", paced_runs))
    finally:
        server.kill()

    markdown = "\n".join([
        f"{cpu_name()}.",
        "",
        "Instant model (the cost of the Python side alone):",
        "",
        *instant,
        "",
        "Paced model (100 pieces, 10 ms apart):",
        "",
        *paced,
    ])  # fmt: skip
    print(markdown)
    if args.runs == parser.get_default("runs"):
        write_into_readme("stream", markdown)


if __name__ == "__main__":
    main()
