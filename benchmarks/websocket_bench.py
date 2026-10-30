"""Compare the Responses API over HTTP and over WebSocket, in pyox-agents and in the OpenAI Agents SDK.

    uv run websocket_bench.py                        # full run, writes the table in README.md (~15 min)
    uv run websocket_bench.py --runs 50 --only "pyox-agents websocket" "openai-agents websocket"  # README unchanged
    uv run websocket_bench.py --model-delay-ms 50    # the fake model waits 50 ms before each answer

    # Opt-in, against api.openai.com (costs tokens, README unchanged):
    OPENAI_API_KEY=sk-... uv run websocket_bench.py --real-openai --model gpt-5-mini --runs 5 --concurrency 1

Steps:
1. Start the fake Responses API server (`responses_server.py`), unless `--real-openai`.
2. Each agent on each tool loop (1, 5 and 20 tool calls, one per turn): `measure.py` in a new process. The
   server's counters, read before and after, give the bytes each run uploads and the WebSocket connections opened.
3. Print the tables; a full run also writes them in README.md, between the "websocket" markers.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

from libraries import RESPONSES_TRANSPORTS
from measure import TOOL_LOOP_SCENARIOS
from run_all import HERE, cpu_name, markdown_table, results_table, write_into_readme

PORT = 4020
FAKE_BASE_URL = f"http://127.0.0.1:{PORT}/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
WARMUP_RUNS = 10  # done by measure.py before measuring


def start_fake_server(model_delay_ms: float) -> subprocess.Popen:
    command = [sys.executable, "-m", "uvicorn", "responses_server:app", "--port", str(PORT), "--log-level", "warning"]
    server = subprocess.Popen(command, cwd=HERE, env={**os.environ, "MODEL_DELAY_MS": str(model_delay_ms)})
    for _ in range(100):
        try:
            server_stats()
            return server
        except OSError:
            time.sleep(0.1)
    server.kill()
    raise RuntimeError("The fake Responses server did not start")


def server_stats() -> dict[str, int]:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/stats") as response:
        return json.load(response)


def measure(agent: str, scenario: str, args: argparse.Namespace) -> dict:
    base_url = OPENAI_BASE_URL if args.real_openai else FAKE_BASE_URL
    command = [sys.executable, "measure.py", agent, scenario, "--base-url", base_url]
    command += ["--runs", str(args.runs), "--concurrency", str(args.concurrency)]
    environment = dict(os.environ)
    if args.real_openai:
        command.append("--any-answer")
        environment |= {"BENCHMARK_API_KEY": os.environ["OPENAI_API_KEY"], "BENCHMARK_MODEL": args.model}

    print(f"Measuring {agent} / {scenario}...", flush=True)
    before = {} if args.real_openai else server_stats()
    output = subprocess.run(command, cwd=HERE, env=environment, check=True, stdout=subprocess.PIPE, text=True).stdout
    result = json.loads(output)
    if not args.real_openai:
        after = server_stats()
        all_runs = WARMUP_RUNS + 2 * args.runs  # warm-up, sequential and concurrent runs
        result["upload_kb_per_run"] = (after["request_bytes"] - before["request_bytes"]) / all_runs / 1024
        result["websocket_connections"] = after["websocket_connections"] - before["websocket_connections"]
    return result


def results_markdown(results: list[dict], args: argparse.Namespace) -> str:
    where = "api.openai.com, model " + args.model if args.real_openai else "the fake server"
    lines = [
        f"Python {sys.version.split()[0]}, {cpu_name()}, {where}, model delay {args.model_delay_ms:g} ms per turn."
        f" {args.runs} runs per measurement, {args.concurrency} at a time for throughput.",
        "",
        "Latency p50 of one run, by number of tool calls (one per turn). Fastest first.",
        "",
    ]
    scenarios = list(TOOL_LOOP_SCENARIOS)
    longest = scenarios[-1]
    by_key = {(r["agent"], r["scenario"]): r for r in results}
    agents = sorted(dict.fromkeys(r["agent"] for r in results), key=lambda a: by_key[a, longest]["latency_p50_ms"])
    headers = ["Agent", *(f"`{scenario}`" for scenario in scenarios), f"Throughput, `{longest}`"]
    headers += [f"Upload per run, `{longest}`", f"Connections, `{longest}`"]
    rows = []
    for agent in agents:
        longest_result = by_key[agent, longest]
        rows.append(
            [
                agent,
                *(f"{by_key[agent, scenario]['latency_p50_ms']:.1f} ms" for scenario in scenarios),
                f"{longest_result['runs_per_second']:.0f} runs/s",
                f"{longest_result['upload_kb_per_run']:.1f} KB" if "upload_kb_per_run" in longest_result else "-",
                str(longest_result.get("websocket_connections", "-")),
            ]
        )
    lines += markdown_table(headers, rows)
    lines += ["", "<details><summary>All the numbers</summary>"]
    for scenario in scenarios:
        scenario_results = [r for r in results if r["scenario"] == scenario]
        lines += ["", f"`{scenario}`", "", *results_table("Agent", scenario_results)]
    lines += ["", "</details>"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--model-delay-ms", type=float, default=0, help="fake model's wait before each answer")
    parser.add_argument("--only", nargs="+", choices=RESPONSES_TRANSPORTS, help="measure only these (README unchanged)")
    parser.add_argument("--real-openai", action="store_true", help="use api.openai.com and $OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-5-mini", help="model for --real-openai")
    args = parser.parse_args()
    agents = args.only or list(RESPONSES_TRANSPORTS)

    server = None if args.real_openai else start_fake_server(args.model_delay_ms)
    try:
        results = [measure(agent, scenario, args) for scenario in TOOL_LOOP_SCENARIOS for agent in agents]
    finally:
        if server is not None:
            server.terminate()
            server.wait()

    markdown = results_markdown(results, args)
    print(markdown)
    if not (args.only or args.real_openai or args.model_delay_ms):
        (HERE / "websocket_results.json").write_text(json.dumps(results, indent=2) + "\n")
        write_into_readme("websocket", markdown)


if __name__ == "__main__":
    main()
