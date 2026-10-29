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

from libraries import LIBRARIES, PACKAGES, TEXT_CHUNKS
from measure import SCENARIOS

HERE = Path(__file__).parent
PORT = 4010
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
AIMOCK = "@copilotkit/aimock@1.46.0"
BASELINE = "pyox-agents"  # the other libraries are compared with this one
SCENARIO_DESCRIPTIONS = {
    "text": "one long streamed answer",
    "tool": "one tool call, then the answer",
    "parallel_tools": "five tool calls at once, then the answer",
}
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


def measure_startup(libraries: list[str]) -> dict[str, dict[str, float]]:
    """Time and memory to start Python, import the library and create an agent (hyperfine 2)."""
    commands = []
    for library in libraries:
        code = f"from libraries import LIBRARIES; LIBRARIES['{library}']('{BASE_URL}')"
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
    libraries = list(startup)
    versions = ", ".join(f"{library} {version(PACKAGES[library])}" for library in libraries)
    lines = [
        f"Python {platform.python_version()}, {platform.system()} {platform.machine()} ({cpu_name()})."
        f" {runs} runs per measurement, {concurrency} at a time for throughput. Versions: {versions}.",
        "",
        "### Overview",
        "",
        "Latency p50 of one run, fastest first; in parentheses, how many times the pyox-agents time it is."
        " Lower is better.",
        "",
    ]
    latency = {(r["agent"], r["scenario"]): r["latency_p50_ms"] for r in results}
    scenarios = list(dict.fromkeys(r["scenario"] for r in results))
    rows = [
        [library, f"{startup[library]['time_ms']:.0f} ms"]
        + [f"{latency[library, s]:.1f} ms ({latency[library, s] / latency[BASELINE, s]:.1f}x)" for s in scenarios]
        for library in sorted(libraries, key=lambda library: latency[library, "tool"])
    ]
    lines += markdown_table(["Library", "Startup", *scenarios], rows)
    for scenario in scenarios:
        description = SCENARIO_DESCRIPTIONS[scenario]
        lines += [
            "",
            f"### `{scenario}`: {description}",
            "",
            *results_table("Library", [r for r in results if r["scenario"] == scenario]),
        ]
    lines += ["", "### Startup", ""]
    rows = [
        [library, f"{startup[library]['time_ms']:.0f} ms", f"{startup[library]['memory_mb']:.0f} MB"]
        for library in sorted(libraries, key=lambda library: startup[library]["time_ms"])
    ]
    lines += markdown_table(["Library", "Startup", "Memory after startup"], rows)
    return "\n".join(lines)


def results_table(name_column: str, results: list[dict]) -> list[str]:
    """One row per agent, fastest first."""
    headers = [
        name_column,
        "Latency p50",
        "Latency p95",
        "First text",
        "CPU per run",
        "Throughput",
        "Failed",
        "Peak memory",
    ]
    rows = [
        [
            r["agent"],
            f"{r['latency_p50_ms']:.1f} ms",
            f"{r['latency_p95_ms']:.1f} ms",
            f"{r['first_text_p50_ms']:.1f} ms",
            f"{r['cpu_ms_per_run']:.1f} ms",
            f"{r['runs_per_second']:.0f} runs/s",
            str(r["failed_runs"]),
            f"{r['memory_peak_mb']:.0f} MB",
        ]
        for r in sorted(results, key=lambda r: r["latency_p50_ms"])
    ]
    return markdown_table(headers, rows)


def markdown_table(headers: list[str], rows: list[list[str]], text_columns: int = 1) -> list[str]:
    """A Markdown table: the first `text_columns` aligned left, the others (numbers) right."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * text_columns + "--:|" * (len(headers) - text_columns)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def write_into_readme(section: str, markdown: str) -> None:
    readme = HERE / "README.md"
    start, end = f"<!-- {section}:start -->", f"<!-- {section}:end -->"
    before, rest = readme.read_text().split(start)
    after = rest.split(end)[1]
    readme.write_text(f"{before}{start}\n{markdown}\n{end}{after}")


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--only", nargs="+", choices=LIBRARIES, help="measure only these libraries (README unchanged)")
    args = parser.parse_args()
    libraries = args.only or list(LIBRARIES)
    if BASELINE not in libraries:
        libraries.insert(0, BASELINE)

    server = start_mock_server()
    try:
        startup = measure_startup(libraries)
        results = [
            measure(library, scenario, args.runs, args.concurrency) for scenario in SCENARIOS for library in libraries
        ]
        text_chunks = [] if args.only else [measure(mode, "text", args.runs, args.concurrency) for mode in TEXT_CHUNKS]
    finally:
        stop_mock_server(server)

    markdown = libraries_markdown(startup, results, args.runs, args.concurrency)
    if args.only:
        print(markdown)
        return
    all_results = {"startup": startup, "libraries": results, "text_chunks": text_chunks}
    (HERE / "results.json").write_text(json.dumps(all_results, indent=2) + "\n")
    write_into_readme("results", markdown)
    write_into_readme("text-chunks", "\n".join(results_table("Mode", text_chunks)))
    print((HERE / "README.md").read_text())


if __name__ == "__main__":
    main()
