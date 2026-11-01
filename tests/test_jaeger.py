"""End to end: a traced run sent over OTLP to a real Jaeger, then read back from Jaeger's query API.

It checks what the in-memory tests cannot: that the spans survive OTLP encoding, and that flushing
the provider (as docs/telemetry.md tells apps to) sends them. CI runs it against a Jaeger service
container (the `jaeger` job in .github/workflows/ci.yml). Skipped unless `JAEGER_URL` is set. Locally:

    docker run --rm -p 4318:4318 -p 16686:16686 jaegertracing/jaeger:2.20.0
    JAEGER_URL=http://localhost:16686 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \\
        uv run --with opentelemetry-exporter-otlp-proto-http pytest tests/test_jaeger.py
"""

import asyncio
import importlib
import json
import os
import urllib.request
import uuid
from typing import Any

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from conftest import MakeAgent
from factories import text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import Instrumentation, tool

JAEGER_URL = os.environ.get("JAEGER_URL")

pytestmark = pytest.mark.skipif(not JAEGER_URL, reason="needs a Jaeger: set JAEGER_URL (see this file's docstring)")


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"Sunny in {city}"


def read_traces(service: str) -> list[dict[str, Any]]:
    with urllib.request.urlopen(f"{JAEGER_URL}/api/traces?service={service}") as response:
        return json.load(response)["data"]


async def test_a_traced_run_reaches_jaeger(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    # Not a dev dependency: the CI job adds it (`uv run --with ...`). Missing, the test fails rather than skips.
    exporter_module = importlib.import_module("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    service = f"pyox-e2e-{uuid.uuid4().hex[:8]}"  # a new service per run: only this test's trace
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    # Reads OTEL_EXPORTER_OTLP_ENDPOINT, like an app's exporter would.
    provider.add_span_processor(BatchSpanProcessor(exporter_module.OTLPSpanExporter()))
    fake_openai.queue(tool_calls_response(tool_call("get_weather", city="Vigo")), text_response("Sunny"))
    agent = make_agent(name="weather", tools=[get_weather], instrument=Instrumentation(tracer_provider=provider))

    await agent.run("Weather in Vigo?")
    provider.force_flush()  # what an app does before it exits (or `provider.shutdown()`)

    traces: list[dict[str, Any]] = []
    for _ in range(50):  # Jaeger stores what it receives a moment later
        traces = await asyncio.to_thread(read_traces, service)
        if traces and len(traces[0]["spans"]) == 4:
            break
        await asyncio.sleep(0.2)
    [trace] = traces
    assert trace["processes"][trace["spans"][0]["processID"]]["serviceName"] == service
    spans = {span["operationName"]: span for span in trace["spans"]}
    assert set(spans) == {"invoke_agent weather", "chat test-model", "execute_tool get_weather"}

    run_id = spans["invoke_agent weather"]["spanID"]
    for name in ("chat test-model", "execute_tool get_weather"):
        [parent] = spans[name]["references"]
        assert (parent["refType"], parent["spanID"]) == ("CHILD_OF", run_id)
    # Attribute types survive the trip: numbers stay numbers.
    tags = {tag["key"]: tag for tag in spans["chat test-model"]["tags"]}
    assert (tags["gen_ai.usage.input_tokens"]["type"], tags["gen_ai.usage.input_tokens"]["value"]) == ("int64", 10)
    assert tags["gen_ai.operation.name"]["value"] == "chat"
    assert tags["span.kind"]["value"] == "client"
