"""Load-test each agent behind a FastAPI streaming endpoint and put the results in README.md.

    uv run load_test.py                  # needs Node.js (for npx) and k6
    uv run load_test.py --duration 10s   # quicker, less precise

For each library, scenario and number of concurrent users:
1. Start `server.py` (uvicorn, one process) with that library, and warm it up.
2. k6 runs `load.js`: that many users send requests in a loop, each reading the whole streamed answer.
3. Ask the server for its peak memory and CPU time (`GET /stats`), then stop it.
Everything runs against the same fake OpenAI server as `run_all.py`.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from agents import LIBRARIES
from measure import SCENARIOS
from run_all import BASE_URL, HERE, cpu_name, start_mock_server, stop_mock_server, write_into_readme

SERVER_PORT = 8000
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
CONCURRENT_USERS = [1, 10, 50, 100]
LOAD_SCENARIOS = ["text", "tool"]


def start_server(library: str) -> subprocess.Popen:
    command = [sys.executable, "-m", "uvicorn", "server:app", "--port", str(SERVER_PORT), "--log-level", "warning"]
    server = subprocess.Popen(command, cwd=HERE, env={**os.environ, "AGENT": library, "MODEL_URL": BASE_URL})
    for _ in range(100):
        try:
            server_stats()
            return server
        except OSError:
            time.sleep(0.1)
    server.kill()
    raise RuntimeError("The FastAPI server did not start")


def server_stats() -> dict[str, float]:
    with urllib.request.urlopen(f"{SERVER_URL}/stats") as response:
        return json.load(response)


def send_chat(prompt: str) -> str:
    request = urllib.request.Request(
        f"{SERVER_URL}/chat", data=json.dumps({"prompt": prompt}).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode()


def load_test(library: str, scenario: str, users: int, duration: str) -> dict:
    print(f"Load testing {library} / {scenario} / {users} users...", flush=True)
    prompt, expected = SCENARIOS[scenario]
    server = start_server(library)
    try:
        for _ in range(10):  # warm up connections and lazy imports
            send_chat(prompt)
        cpu_before = server_stats()["cpu_seconds"]

        with tempfile.NamedTemporaryFile(suffix=".json") as summary_file:
            settings = {"URL": f"{SERVER_URL}/chat", "PROMPT": prompt, "EXPECTED": expected, "VUS": users}
            settings |= {"DURATION": duration, "SUMMARY_FILE": summary_file.name}
            variables = [part for name, value in settings.items() for part in ("-e", f"{name}={value}")]
            subprocess.run(["k6", "run", "--quiet", *variables, "load.js"], cwd=HERE, check=True)
            metrics = json.loads(Path(summary_file.name).read_text())["metrics"]

        stats = server_stats()
    finally:
        server.terminate()
        server.wait()

    requests = metrics["iterations"]["values"]["count"]
    duration_ms = metrics["http_req_duration"]["values"]
    first_byte_ms = metrics["http_req_waiting"]["values"]
    return {
        "library": library,
        "scenario": scenario,
        "users": users,
        "requests_per_second": metrics["iterations"]["values"]["rate"],
        "first_byte_p50_ms": first_byte_ms["p(50)"],
        "first_byte_p99_ms": first_byte_ms["p(99)"],
        "latency_p50_ms": duration_ms["p(50)"],
        "latency_p90_ms": duration_ms["p(90)"],
        "latency_p99_ms": duration_ms["p(99)"],
        "failed_requests": metrics["checks"]["values"]["fails"],
        "cpu_ms_per_request": (stats["cpu_seconds"] - cpu_before) * 1000 / requests,
        "memory_peak_mb": stats["memory_peak_mb"],
    }


def results_markdown(results: list[dict], duration: str) -> str:
    lines = [
        f"Python {sys.version.split()[0]}, one uvicorn process, {cpu_name()}. Each row: {duration} of load.",
        "",
        "| Scenario | Users | Library | Requests/s | First byte p50 | p99 | Latency p50 | p90 | p99 | Failed"
        " | Server CPU per request | Server peak memory |",
        "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        lines.append(
            f"| {r['scenario']} | {r['users']} | {r['library']} | {r['requests_per_second']:.0f}"
            f" | {r['first_byte_p50_ms']:.1f} ms | {r['first_byte_p99_ms']:.1f} ms"
            f" | {r['latency_p50_ms']:.1f} ms | {r['latency_p90_ms']:.1f} ms | {r['latency_p99_ms']:.1f} ms"
            f" | {r['failed_requests']} | {r['cpu_ms_per_request']:.1f} ms | {r['memory_peak_mb']:.0f} MB |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", default="20s", help="how long each load test runs (k6 format)")
    args = parser.parse_args()

    mock_server = start_mock_server()
    try:
        results = [
            load_test(library, scenario, users, args.duration)
            for scenario in LOAD_SCENARIOS
            for users in CONCURRENT_USERS
            for library in LIBRARIES
        ]
    finally:
        stop_mock_server(mock_server)

    (HERE / "load_test_results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_into_readme("server", results_markdown(results, args.duration))
    print(results_markdown(results, args.duration))


if __name__ == "__main__":
    main()
