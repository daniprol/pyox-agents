"""Run every benchmark and put the results in README.md.

    uv run run_all.py              # needs Node.js (for npx) and hyperfine
    uv run run_all.py --runs 100   # quicker, less precise

Steps:
1. Start the fake OpenAI server (aimock) with the answers in fixtures.json.
2. Startup time: hyperfine runs "import the library and create an agent" many times.
3. For each scenario and library: `measure.py` in a new process (latency, CPU, throughput, memory).
4. Write the tables in README.md, between the "results" markers.
"""

import argparse
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from agents import LIBRARIES
from measure import SCENARIOS

HERE = Path(__file__).parent
PORT = 4010
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
AIMOCK = "@copilotkit/aimock@1.46.0"
# `-c 4`: about one token (~4 characters) per streamed chunk, like a real model.
PACKAGES = {"pyox-agents": "pyox-agents", "pydantic-ai": "pydantic-ai-slim"}  # library -> installed package
MOCK_SERVER = ["npx", "--yes", f"--package={AIMOCK}", "llmock", "-f", "fixtures.json", "-p", str(PORT), "-c", "4"]


def start_mock_server() -> subprocess.Popen:
    # A new process group, so `stop_mock_server` also stops the node process started by npx.
    server = subprocess.Popen([*MOCK_SERVER, "--log-level", "warn"], cwd=HERE, start_new_session=True)
    for _ in range(300):  # npx may need to download it the first time
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            return server
        except OSError:
            time.sleep(0.2)
    stop_mock_server(server)
    raise RuntimeError("The mock server did not start")


def stop_mock_server(server: subprocess.Popen) -> None:
    os.killpg(server.pid, signal.SIGTERM)


def measure_startup() -> dict[str, dict[str, float]]:
    """Time and memory to start Python, import the library and create an agent (hyperfine 2)."""
    commands = []
    for library in LIBRARIES:
        code = f"from agents import LIBRARIES; LIBRARIES['{library}']('{BASE_URL}')"
        commands += ["--command-name", library, f'{sys.executable} -c "{code}"']
    with tempfile.NamedTemporaryFile(suffix=".json") as output:
        hyperfine = ["hyperfine", "--warmup", "3", "--runs", "30", "--export-json", output.name]
        subprocess.run([*hyperfine, *commands], cwd=HERE, check=True)
        results = json.loads(Path(output.name).read_text())["results"]
    return {
        result["name"]: {
            "time_ms": result["summary"]["time_wall_clock"]["mean"] * 1000,
            "memory_mb": result["summary"]["memory_peak_resident"]["mean"] / 1024 / 1024,
        }
        for result in results
    }


def measure(library: str, scenario: str, runs: int, concurrency: int) -> dict:
    command = [sys.executable, "measure.py", library, scenario, "--base-url", BASE_URL]
    command += ["--runs", str(runs), "--concurrency", str(concurrency)]
    print(f"Measuring {library} / {scenario}...", flush=True)
    output = subprocess.run(command, cwd=HERE, check=True, capture_output=True, text=True).stdout
    return json.loads(output)


def results_markdown(startup: dict[str, dict[str, float]], results: list[dict], runs: int, concurrency: int) -> str:
    versions = ", ".join(f"{library} {version(package)}" for library, package in PACKAGES.items())
    lines = [
        f"Python {platform.python_version()}, {versions}, {platform.system()} {platform.machine()}"
        f" ({_cpu_name()}). {runs} runs per measurement, {concurrency} at a time for throughput.",
        "",
        "| Library | Startup | Memory after startup |",
        "|---|--:|--:|",
    ]
    for library in LIBRARIES:
        lines.append(f"| {library} | {startup[library]['time_ms']:.0f} ms | {startup[library]['memory_mb']:.0f} MB |")
    lines += [
        "",
        "| Scenario | Library | Latency p50 | Latency p95 | CPU per run | Throughput | Peak memory |",
        "|---|---|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        lines.append(
            f"| {r['scenario']} | {r['library']} | {r['latency_p50_ms']:.1f} ms | {r['latency_p95_ms']:.1f} ms"
            f" | {r['cpu_ms_per_run']:.1f} ms | {r['runs_per_second']:.0f} runs/s | {r['memory_peak_mb']:.0f} MB |"
        )
    return "\n".join(lines)


def write_into_readme(markdown: str) -> None:
    readme = HERE / "README.md"
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    before, rest = readme.read_text().split(start)
    after = rest.split(end)[1]
    readme.write_text(f"{before}{start}\n{markdown}\n{end}{after}")


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=100)
    args = parser.parse_args()

    server = start_mock_server()
    try:
        startup = measure_startup()
        results = [
            measure(library, scenario, args.runs, args.concurrency) for scenario in SCENARIOS for library in LIBRARIES
        ]
    finally:
        stop_mock_server(server)

    (HERE / "results.json").write_text(json.dumps({"startup": startup, "results": results}, indent=2) + "\n")
    markdown = results_markdown(startup, results, args.runs, args.concurrency)
    write_into_readme(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
