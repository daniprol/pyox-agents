"""Draw the benchmark chart shown at the top of the README, from results.json.

    uv run chart.py   # after run_all.py; writes docs/assets/benchmark-{light,dark}.svg

Three panels for the `tool` scenario (one tool call, then the answer): latency, throughput and peak memory.
Plain SVG with no dependencies, one file per GitHub theme.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
OUTPUT = HERE.parent / "docs" / "assets"
BASELINE = "pyox-agents"

PANELS = [
    # (title, unit line, result key, number format)
    ("Latency of one run", "p50, ms (lower is better)", "latency_p50_ms", "{:.1f}"),
    ("Throughput", "runs/s, 100 at once (higher is better)", "runs_per_second", "{:.0f}"),
    ("Peak memory", "MB (lower is better)", "memory_peak_mb", "{:.0f}"),
]

THEMES = {
    "light": {"text": "#0F172A", "muted": "#64748B", "bar": "#CBD5E1", "highlight": "#EA580C", "rule": "#E2E8F0"},
    "dark": {"text": "#E6EDF3", "muted": "#8B949E", "bar": "#3D444D", "highlight": "#F97316", "rule": "#30363D"},
}

WIDTH = 960
LABEL_WIDTH = 132  # library names on the left
PANEL_GAP = 28
TOP = 66  # room for the panel titles
ROW = 30
BAR = 16
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def load_tool_results() -> list[dict]:
    results = json.loads((HERE / "results.json").read_text())["libraries"]
    rows = [row for row in results if row["scenario"] == "tool"]
    return sorted(rows, key=lambda row: row["latency_p50_ms"])  # fastest first


def draw(rows: list[dict], colors: dict[str, str]) -> str:
    panel_width = (WIDTH - LABEL_WIDTH - PANEL_GAP * (len(PANELS) - 1)) / len(PANELS)
    value_room = 44  # space after the longest bar for its number
    height = TOP + ROW * len(rows) + 8
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" height="{height}" '
        f'font-family="{FONT}" role="img" aria-label="Benchmark: pyox-agents against seven agent frameworks">'
    ]
    for index, row in enumerate(rows):
        y = TOP + index * ROW + ROW / 2
        is_baseline = row["agent"] == BASELINE
        weight = "600" if is_baseline else "400"
        fill = colors["text"] if is_baseline else colors["muted"]
        parts.append(
            f'<text x="{LABEL_WIDTH - 12}" y="{y + 4.5:.1f}" text-anchor="end" font-size="14" '
            f'font-weight="{weight}" fill="{fill}">{row["agent"]}</text>'
        )
    for panel, (title, unit, key, number_format) in enumerate(PANELS):
        left = LABEL_WIDTH + panel * (panel_width + PANEL_GAP)
        largest = max(row[key] for row in rows)
        scale = (panel_width - value_room) / largest
        parts.append(
            f'<text x="{left:.1f}" y="20" font-size="15" font-weight="600" fill="{colors["text"]}">{title}</text>'
        )
        parts.append(f'<text x="{left:.1f}" y="40" font-size="13" fill="{colors["muted"]}">{unit}</text>')
        parts.append(
            f'<line x1="{left:.1f}" y1="{TOP - 6}" x2="{left:.1f}" y2="{height - 4}" stroke="{colors["rule"]}"/>'
        )
        for index, row in enumerate(rows):
            is_baseline = row["agent"] == BASELINE
            bar_width = max(row[key] * scale, 2)
            y = TOP + index * ROW + (ROW - BAR) / 2
            fill = colors["highlight"] if is_baseline else colors["bar"]
            parts.append(
                f'<rect x="{left:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{BAR}" rx="3" fill="{fill}"/>'
            )
            weight = "600" if is_baseline else "400"
            text_fill = colors["text"] if is_baseline else colors["muted"]
            parts.append(
                f'<text x="{left + bar_width + 6:.1f}" y="{y + BAR / 2 + 4.5:.1f}" font-size="13" '
                f'font-weight="{weight}" fill="{text_fill}">{number_format.format(row[key])}</text>'
            )
    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> None:
    rows = load_tool_results()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for theme, colors in THEMES.items():
        path = OUTPUT / f"benchmark-{theme}.svg"
        path.write_text(draw(rows, colors))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
