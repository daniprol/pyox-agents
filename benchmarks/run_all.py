"""Run every benchmark and put the results in README.md.

    uv run run_all.py              # needs Node.js (for npx) and hyperfine
    uv run run_all.py --runs 100   # quicker, less precise

Steps:
1. Start the fake OpenAI server (aimock) with the answers in fixtures.json.
2. Startup time: hyperfine runs "import the library and create an agent" many times.
3. Each library on each scenario: `measure.py` in a new process (latency, CPU, throughput, memory).
4. pyox-agents with each `text_chunks` mode on the long streamed answer, the same way.
5. Write the tables in README.md, between the "results" and "text-chunks" markers.
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

from agents import LIBRARIES, TEXT_CHUNKS
from measure import SCENARIOS

HERE = Path(__file__).parent
PORT = 4010
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
AIMOCK = "@copilotkit/aimock@1.46.0"
PACKAGES = {"pyox-agents": "pyox-agents", "pydantic-ai": "pydantic-ai-slim"}  # library -> installed package
# `-c 4`: about one token (~4 characters) per streamed chunk, like a real model.
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


def measure(agent: str, scenario: str, runs: int, concurrency: int) -> dict:
    command = [sys.executable, "measure.py", agent, scenario, "--base-url", BASE_URL]
    command += ["--runs", str(runs), "--concurrency", str(concurrency)]
    print(f"Measuring {agent} / {scenario}...", flush=True)
    output = subprocess.run(command, cwd=HERE, check=True, stdout=subprocess.PIPE, text=True).stdout
    return json.loads(output)


def libraries_markdown(startup: dict[str, dict[str, float]], results: list[dict], runs: int, concurrency: int) -> str:
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
    lines += ["", *results_table("Scenario", "Library", results)]
    return "\n".join(lines)


def results_table(first_column: str, second_column: str, results: list[dict]) -> list[str]:
    lines = [
        f"| {first_column} | {second_column} | First text | Latency p50 | Latency p95 | CPU per run | Throughput"
        " | Failed | Peak memory |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in results:
        lines.append(
            f"| {r['scenario']} | {r['agent']} | {r['first_text_p50_ms']:.1f} ms | {r['latency_p50_ms']:.1f} ms"
            f" | {r['latency_p95_ms']:.1f} ms | {r['cpu_ms_per_run']:.1f} ms | {r['runs_per_second']:.0f} runs/s"
            f" | {r['failed_runs']} | {r['memory_peak_mb']:.0f} MB |"
        )
    return lines


def write_into_readme(section: str, markdown: str) -> None:
    readme = HERE / "README.md"
    start, end = f"<!-- {section}:start -->", f"<!-- {section}:end -->"
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
        libraries = [
            measure(library, scenario, args.runs, args.concurrency) for scenario in SCENARIOS for library in LIBRARIES
        ]
        text_chunks = [measure(mode, "text", args.runs, args.concurrency) for mode in TEXT_CHUNKS]
    finally:
        stop_mock_server(server)

    all_results = {"startup": startup, "libraries": libraries, "text_chunks": text_chunks}
    (HERE / "results.json").write_text(json.dumps(all_results, indent=2) + "\n")
    write_into_readme("results", libraries_markdown(startup, libraries, args.runs, args.concurrency))
    write_into_readme("text-chunks", "\n".join(results_table("Scenario", "Mode", text_chunks)))
    print((HERE / "README.md").read_text())


if __name__ == "__main__":
    main()
