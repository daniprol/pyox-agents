import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from fake_openai import FakeOpenAI
from pyox_agents import Agent, Client, Instrumentation

MakeAgent = Callable[..., Agent]


@pytest.fixture
def fake_openai() -> Iterator[FakeOpenAI]:
    server = FakeOpenAI()
    yield server
    server.close()


@pytest.fixture
def client(fake_openai: FakeOpenAI) -> Client:
    return Client(api_key="test-key", base_url=fake_openai.base_url, headers={"X-Test": "yes"})


@pytest.fixture
def websocket_client(fake_openai: FakeOpenAI) -> Client:
    """Like `client`, sending Responses API requests over WebSocket."""
    return Client(api_key="test-key", base_url=fake_openai.base_url, headers={"X-Test": "yes"}, transport="websocket")


@pytest.fixture
def model() -> str:
    return "test-model"


@pytest.fixture
def make_agent(client: Client, model: str) -> MakeAgent:
    """Factory for agents connected to the fake server: `make_agent(tools=[...])`.

    The live tests (tests/test_live.py) override `client` and `model` to reuse it against a real model.
    """

    def make(**options: Any) -> Agent:
        options.setdefault("client", client)
        options.setdefault("model", model)
        return Agent(**options)

    return make


@dataclass
class Tracing:
    """A real OpenTelemetry SDK that keeps spans and metrics in memory (see the `tracing` fixture)."""

    instrumentation: Instrumentation  # give it to an agent: `make_agent(instrument=tracing.instrumentation)`
    tracer: Tracer  # for the test's own spans (in tools, hooks...)
    exporter: InMemorySpanExporter
    metrics: InMemoryMetricReader

    @property
    def spans(self) -> tuple[ReadableSpan, ...]:
        """The spans that ended, in the order they ended."""
        return self.exporter.get_finished_spans()

    def span(self, name: str) -> ReadableSpan:
        """The one ended span named `name`."""
        [span] = [span for span in self.spans if span.name == name]
        return span


@pytest.fixture
def tracing() -> Tracing:
    """Providers made for one test (never the global ones), so tests can run in parallel."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    metrics = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metrics])
    instrumentation = Instrumentation(tracer_provider=tracer_provider, meter_provider=meter_provider)
    return Tracing(instrumentation, tracer_provider.get_tracer("tests"), exporter, metrics)


PROTOCOL_CLIENTS = Path(__file__).parent / "protocol_clients"

CheckStream = Callable[..., Any]


@pytest.fixture
def protocol_client() -> CheckStream:
    """Checks a stream with the official JavaScript client: `protocol_client("ai-sdk", sse_text)`.

    Returns what the client built from it (see `tests/protocol_clients/check.mjs`). Needs Node
    and `npm ci --prefix tests/protocol_clients`; skipped without them, except on CI.
    """
    if shutil.which("node") is None or not (PROTOCOL_CLIENTS / "node_modules").exists():
        if os.environ.get("CI"):
            pytest.fail("CI must check streams with the official clients: run `npm ci --prefix tests/protocol_clients`")
        pytest.skip("needs node and `npm ci --prefix tests/protocol_clients`")

    def check(protocol: str, sse: str, message: dict[str, Any] | None = None) -> Any:
        result = subprocess.run(
            ["node", "check.mjs", protocol],
            input=json.dumps({"sse": sse, "message": message}),
            cwd=PROTOCOL_CLIENTS,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    return check
