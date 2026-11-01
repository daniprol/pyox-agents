"""OpenTelemetry tracing (`pyox_agents.instrument()`, `Agent(instrument=...)`), with the real SDK in memory."""

import asyncio
import dataclasses
import json
import sys
from typing import Any
from urllib.parse import urlsplit

import pytest
from opentelemetry.sdk.metrics.export import HistogramDataPoint
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import pyox_agents.telemetry
from conftest import MakeAgent, Tracing
from factories import (
    collect_events,
    error_response,
    read_text,
    responses_answer,
    run_finished,
    span_tree,
    text_response,
    tool_call,
    tool_calls_response,
)
from fake_openai import FakeOpenAI, StreamResponse
from pyox_agents import (
    ApiError,
    Approve,
    Hooks,
    Instrumentation,
    ModelRequest,
    RunContext,
    TextDelta,
    ToolCall,
    instrument,
    tool,
)
from pyox_agents.telemetry import recorder_for


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"Sunny in {city}"


@tool(requires_approval=True)
def book(flight: str) -> str:
    """Book a flight."""
    return f"booked {flight}"


def attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def events(span: ReadableSpan) -> list[tuple[str, dict[str, Any]]]:
    return [(event.name, dict(event.attributes or {})) for event in span.events]


def parent_of(span: ReadableSpan) -> int | None:
    return span.parent.span_id if span.parent else None


def span_id(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.span_id


# The span tree and its attributes -----------------------------------------------------------------


async def test_a_run_is_a_tree_of_spans(fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("get_weather", city="Vigo")),
        StreamResponse(text_response("Sunny").chunks, headers={"x-request-id": "req_42"}),
    )
    agent = make_agent(name="weather", tools=[get_weather], instrument=tracing.instrumentation)

    await agent.run("Weather in Vigo?")

    assert span_tree(tracing.spans) == (
        "invoke_agent weather\n  chat test-model\n  execute_tool get_weather\n  chat test-model"
    )
    run = tracing.span("invoke_agent weather")
    assert parent_of(run) is None
    assert attributes(run) == {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.provider.name": "openai",
        "gen_ai.system": "openai",
        "gen_ai.agent.name": "weather",
        "gen_ai.request.model": "test-model",
        "pyox.run.turns": 2,
        "pyox.run.stop_reason": "end",
        "pyox.usage.input_tokens": 20,
        "pyox.usage.output_tokens": 10,
        "pyox.usage.cached_input_tokens": 0,
        "pyox.usage.reasoning_tokens": 0,
    }

    first_chat, last_chat = [span for span in tracing.spans if span.name == "chat test-model"]
    chat = attributes(last_chat)
    assert chat.pop("gen_ai.response.time_to_first_chunk") > 0
    assert chat == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.system": "openai",
        "gen_ai.request.model": "test-model",
        "gen_ai.response.finish_reasons": ("end",),
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 5,
        "gen_ai.usage.cache_read.input_tokens": 0,
        "gen_ai.usage.reasoning.output_tokens": 0,
        "server.address": "127.0.0.1",
        "server.port": urlsplit(fake_openai.base_url).port,
        "pyox.turn": 2,
        "pyox.http.retries": 0,
        "pyox.openai.request_id": "req_42",
    }
    assert attributes(first_chat)["pyox.turn"] == 1
    assert {first_chat.kind.name, last_chat.kind.name, run.kind.name} == {"CLIENT", "INTERNAL"}

    execute_tool = tracing.span("execute_tool get_weather")
    assert attributes(execute_tool) == {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "get_weather",
        "gen_ai.tool.call.id": "call_0",
        "gen_ai.tool.type": "function",
        "gen_ai.tool.description": "Get the weather in a city.",
        "gen_ai.agent.name": "weather",
        "pyox.tool.kind": "python",
    }
    # Every span keeps the times Rust measured: in order, and inside the run.
    starts = [span.start_time or 0 for span in (run, first_chat, execute_tool, last_chat)]
    assert starts == sorted(starts)
    assert (last_chat.end_time or 0) <= (run.end_time or 0)


async def test_an_agent_without_a_name_and_its_conversation_id(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    class Store:
        def load(self, conversation_id: str) -> list[Any]:
            return []

        def save(self, conversation_id: str, new_messages_json: str) -> None:
            pass

    fake_openai.queue(text_response("Hi"))
    agent = make_agent(history_store=Store(), instrument=tracing.instrumentation)

    await agent.run("Hi", conversation_id="user-42")

    assert span_tree(tracing.spans) == "invoke_agent\n  chat test-model"
    run = attributes(tracing.span("invoke_agent"))
    assert run["gen_ai.conversation.id"] == "user-42"
    assert "gen_ai.agent.name" not in run


async def test_chat_spans_record_the_request_as_sent_and_its_retries(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    def cooler(ctx: RunContext, request: ModelRequest) -> ModelRequest:
        return {**request, "model_parameters": {**request["model_parameters"], "temperature": 0.1}}

    fake_openai.queue(error_response(503, "Busy", headers={"retry-after-ms": "1"}), text_response("Hi"))
    agent = make_agent(
        model_parameters={"temperature": 0.9, "max_completion_tokens": 50, "stop": "END"},
        hooks=[Hooks(before_model_request=cooler)],
        instrument=tracing.instrumentation,
    )

    await agent.run("Hi")

    chat = attributes(tracing.span("chat test-model"))
    assert chat["gen_ai.request.temperature"] == 0.1  # the hook's value: what was sent
    assert chat["gen_ai.request.max_tokens"] == 50
    assert chat["gen_ai.request.stop_sequences"] == ("END",)
    assert chat["pyox.http.retries"] == 1


async def test_responses_api_requests_have_the_response_id(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    fake_openai.queue(responses_answer("Hi", response_id="resp_7"))

    await make_agent(api="responses", instrument=tracing.instrumentation).run("Hi")

    assert attributes(tracing.span("chat test-model"))["gen_ai.response.id"] == "resp_7"


# Parallel tools and the app's own spans -----------------------------------------------------------


async def test_parallel_tools_are_siblings_and_the_app_spans_nest_in_each(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    both_running = asyncio.Barrier(2)

    @tool
    async def search(query: str) -> str:
        """Search."""
        with tracing.tracer.start_as_current_span(f"http {query}"):
            await both_running.wait()  # the two calls really run at the same time
            return "results"

    @tool
    def lookup(key: str) -> str:
        """Look up a key (sync: it runs on a thread)."""
        with tracing.tracer.start_as_current_span(f"db {key}"):
            return "value"

    fake_openai.queue(
        tool_calls_response(
            tool_call("search", query="a"), tool_call("search", query="b"), tool_call("lookup", key="k")
        ),
        text_response("Done"),
    )
    agent = make_agent(name="a", tools=[search, lookup], instrument=tracing.instrumentation)

    await agent.run("Go")

    calls = {attributes(span)["gen_ai.tool.call.id"]: span for span in tracing.spans if "execute_tool" in span.name}
    assert {parent_of(span) for span in calls.values()} == {span_id(tracing.span("invoke_agent a"))}
    assert parent_of(tracing.span("http a")) == span_id(calls["call_0"])
    assert parent_of(tracing.span("http b")) == span_id(calls["call_1"])
    assert parent_of(tracing.span("db k")) == span_id(calls["call_2"])


async def test_hooks_and_approval_functions_run_inside_the_right_span(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    def before_run(ctx: RunContext) -> None:
        tracing.tracer.start_span("before_run hook").end()

    async def before_tool_call(ctx: RunContext, call: ToolCall) -> None:
        tracing.tracer.start_span("before_tool_call hook").end()

    def needs_approval(ctx: RunContext, arguments: dict[str, Any]) -> bool:
        tracing.tracer.start_span("approval function").end()
        return False

    @tool(requires_approval=needs_approval)
    def ping() -> str:
        """Ping."""
        return "pong"

    fake_openai.queue(tool_calls_response(tool_call("ping")), text_response("Done"))
    hooks = Hooks(before_run=before_run, before_tool_call=before_tool_call)
    agent = make_agent(name="a", tools=[ping], hooks=[hooks], instrument=tracing.instrumentation)

    await agent.run("Ping")

    assert span_tree(tracing.spans) == (
        "invoke_agent a\n"
        "  before_run hook\n"
        "  chat test-model\n"
        "  execute_tool ping\n"
        "    before_tool_call hook\n"
        "    approval function\n"
        "  chat test-model"
    )


async def test_a_run_goes_under_the_span_current_at_execute_even_read_in_another_task(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    fake_openai.queue(text_response("Hi"))
    agent = make_agent(name="a", instrument=tracing.instrumentation)

    with tracing.tracer.start_as_current_span("POST /chat"):
        run = agent.execute("Hi")  # like `StreamingResponse(run.ai_sdk())`, read after the handler returned
    text = await asyncio.create_task(read_text(run.ai_sdk()))

    assert '"type":"finish"' in text
    assert span_tree(tracing.spans) == "POST /chat\n  invoke_agent a\n    chat test-model"


# Subagents ----------------------------------------------------------------------------------------


async def test_a_subagent_runs_inside_its_tool_call(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    researcher = make_agent(model="small-model", tools=[get_weather])  # no name: named after its tool
    writer = make_agent(
        name="writer", tools=[researcher.as_tool("research", "Research a topic")], instrument=tracing.instrumentation
    )
    fake_openai.queue(
        tool_calls_response(tool_call("research", task="Weather in Vigo")),
        tool_calls_response(tool_call("get_weather", city="Vigo")),  # the subagent
        text_response("Sunny"),  # the subagent
        text_response("It is sunny"),
    )

    result = await writer.run("Write about Vigo")

    assert span_tree(tracing.spans) == (
        "invoke_agent writer\n"
        "  chat test-model\n"
        "  execute_tool research\n"
        "    invoke_agent research\n"
        "      chat small-model\n"
        "      execute_tool get_weather\n"
        "      chat small-model\n"
        "  chat test-model"
    )
    subagent = attributes(tracing.span("invoke_agent research"))
    assert (subagent["pyox.subagent_id"], subagent["gen_ai.request.model"]) == ("call_0", "small-model")
    assert subagent["pyox.usage.input_tokens"] == 20
    assert attributes(tracing.span("execute_tool research"))["pyox.tool.kind"] == "subagent"
    assert attributes(tracing.span("execute_tool get_weather"))["gen_ai.agent.name"] == "research"
    # The run's totals include the subagent; only chat spans have `gen_ai.usage.*`.
    assert attributes(tracing.span("invoke_agent writer"))["pyox.usage.input_tokens"] == result.usage.input_tokens
    assert sum(attributes(span).get("gen_ai.usage.input_tokens", 0) for span in tracing.spans) == 40


# Errors, stop() and dropped runs ------------------------------------------------------------------


async def test_an_api_error_fails_the_chat_and_run_spans(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    fake_openai.queue(error_response(400, "Bad request", headers={"x-request-id": "req_9"}))

    with pytest.raises(ApiError):
        await make_agent(name="a", instrument=tracing.instrumentation).run("Hi")

    for span in (tracing.span("chat test-model"), tracing.span("invoke_agent a")):
        assert span.status.status_code == StatusCode.ERROR
        assert attributes(span)["error.type"] == "ApiError"
    assert attributes(tracing.span("chat test-model"))["pyox.openai.request_id"] == "req_9"


async def test_a_failing_tool_fails_only_its_span(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    @tool
    def broken() -> str:
        """Fails."""
        raise ValueError("no luck")

    fake_openai.queue(tool_calls_response(tool_call("broken")), text_response("Sorry"))

    await make_agent(name="a", tools=[broken], instrument=tracing.instrumentation).run("Go")

    failed = tracing.span("execute_tool broken")
    assert (failed.status.status_code, attributes(failed)["error.type"]) == (StatusCode.ERROR, "tool_error")
    assert tracing.span("invoke_agent a").status.status_code == StatusCode.UNSET


async def test_stop_ends_the_chat_span_as_stopped(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    chunks = [{"choices": [{"delta": {"content": piece}}]} for piece in ("Once ", "upon ", "a ", "time")]
    fake_openai.queue(StreamResponse(chunks, chunk_delay_seconds=0.1))
    run = make_agent(name="a", instrument=tracing.instrumentation).execute("Tell me a story")

    async for event in run:
        if isinstance(event, TextDelta):
            run.stop()

    assert attributes(tracing.span("chat test-model"))["gen_ai.response.finish_reasons"] == ("stopped",)
    assert attributes(tracing.span("invoke_agent a"))["pyox.run.stop_reason"] == "stopped"


async def test_a_dropped_run_ends_its_spans_as_cancelled(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    started = asyncio.Event()

    @tool
    async def slow() -> str:
        """Takes forever."""
        started.set()
        await asyncio.sleep(60)
        return "never"

    fake_openai.queue(tool_calls_response(tool_call("slow")))
    run = make_agent(name="a", tools=[slow], instrument=tracing.instrumentation).execute("Go")
    async for _ in run:
        await started.wait()
        break  # stop reading: the run is dropped
    del run

    for _ in range(100):  # the aborted loop ends its spans on Rust's side, a moment later
        if len(tracing.spans) == 3:
            break
        await asyncio.sleep(0.01)
    for name in ("execute_tool slow", "invoke_agent a"):
        span = tracing.span(name)
        assert (span.status.status_code, attributes(span)["error.type"]) == (StatusCode.ERROR, "cancelled")


async def test_recorder_errors_never_break_a_run(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenProcessor(SimpleSpanProcessor):
        def on_end(self, span: ReadableSpan) -> None:
            raise RuntimeError("exporter down")

    provider = TracerProvider()
    provider.add_span_processor(BrokenProcessor(InMemorySpanExporter()))
    reported: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", reported.append)  # where Rust reports recorder errors
    fake_openai.queue(text_response("Hi"))

    result = await make_agent(instrument=Instrumentation(tracer_provider=provider)).run("Hi")

    assert result.text == "Hi"
    assert [str(report.exc_value) for report in reported] == ["exporter down"] * 2  # the chat and run spans


# Human in the loop and state ----------------------------------------------------------------------


async def test_a_waiting_call_and_its_resumed_run(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    fake_openai.queue(tool_calls_response(tool_call("book", flight="F1")), text_response("Booked"))
    agent = make_agent(name="travel", tools=[book], instrument=tracing.instrumentation)

    stopped = await agent.run("Book F1")
    first_trace = tracing.spans
    await agent.run(history=stopped.messages, answers={"call_0": Approve()})

    waiting = next(span for span in first_trace if span.name == "execute_tool book")
    assert attributes(waiting)["pyox.tool.waiting"] is True
    assert waiting.status.status_code == StatusCode.UNSET  # waiting for a person is not an error
    assert events(waiting) == [("pyox.input_requested", {"pyox.input.kind": "approval"})]
    first_run = next(span for span in first_trace if span.name == "invoke_agent travel")
    assert attributes(first_run)["pyox.run.pending"] == 1

    # The resumed run is a new trace: the same call id, its reply found in the history.
    resumed = tracing.spans[len(first_trace) :]
    assert span_tree(resumed) == "invoke_agent travel\n  execute_tool book\n  chat test-model"
    resumed_call = next(span for span in resumed if span.name == "execute_tool book")
    assert resumed_call.context.trace_id != waiting.context.trace_id
    assert attributes(resumed_call)["gen_ai.tool.call.id"] == "call_0"
    assert "pyox.tool.waiting" not in attributes(resumed_call)
    assert events(resumed_call) == [
        ("pyox.input_replied", {"pyox.reply.type": "approve", "pyox.reply.source": "history"})
    ]


async def test_live_replies_are_events_inside_the_call(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    def on_request(request: Any) -> Approve:
        tracing.tracer.start_span("on_request handler").end()
        return Approve()

    fake_openai.queue(tool_calls_response(tool_call("book", flight="F1")), text_response("Booked"))
    agent = make_agent(name="a", tools=[book], instrument=tracing.instrumentation)

    await agent.run("Book F1", on_request=on_request)

    call = tracing.span("execute_tool book")
    assert events(call) == [
        ("pyox.input_requested", {"pyox.input.kind": "approval"}),
        ("pyox.input_replied", {"pyox.reply.type": "approve", "pyox.reply.source": "on_request"}),
    ]
    assert parent_of(tracing.span("on_request handler")) == span_id(call)


async def test_state_changes_are_events_with_their_paths(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    @tool
    def add_todo(ctx: RunContext, text: str) -> str:
        """Add a todo."""
        ctx.state["todos"].append(text)
        return "added"

    fake_openai.queue(tool_calls_response(tool_call("add_todo", text="pack")), text_response("Done"))
    agent = make_agent(tools=[add_todo], instrument=tracing.instrumentation)

    await agent.run("Add pack", state={"todos": []})

    assert events(tracing.span("execute_tool add_todo")) == [
        ("pyox.state_changed", {"pyox.state.paths": ("/todos/0",)})
    ]


# Content capture ----------------------------------------------------------------------------------

CONTENT_ATTRIBUTES = {
    "gen_ai.system_instructions",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.tool.definitions",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "pyox.state.delta",
}


def content_attributes(tracing: Tracing) -> set[str]:
    found = {name for span in tracing.spans for name in attributes(span)}
    found |= {name for span in tracing.spans for _, event in events(span) for name in event}
    return found & CONTENT_ATTRIBUTES


async def run_with_content(fake_openai: FakeOpenAI, make_agent: MakeAgent, instrumentation: Instrumentation) -> None:
    @tool
    def add_todo(ctx: RunContext, text: str) -> str:
        """Add a todo."""
        ctx.state["todos"].append(text)
        return "added"

    fake_openai.queue(tool_calls_response(tool_call("add_todo", text="pack")), text_response("Done"))
    agent = make_agent(instructions="Be brief.", tools=[add_todo], instrument=instrumentation)
    await agent.run("Add pack", state={"todos": []})


async def test_content_is_not_recorded_by_default(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    await run_with_content(fake_openai, make_agent, tracing.instrumentation)

    assert content_attributes(tracing) == set()


async def test_content_is_recorded_when_asked(fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing) -> None:
    instrumentation = dataclasses.replace(tracing.instrumentation, capture_content=True)

    await run_with_content(fake_openai, make_agent, instrumentation)

    assert content_attributes(tracing) == CONTENT_ATTRIBUTES
    first_chat, last_chat = [span for span in tracing.spans if span.name == "chat test-model"]
    assert json.loads(attributes(first_chat)["gen_ai.system_instructions"]) == [
        {"type": "text", "content": "Be brief."}
    ]
    assert json.loads(attributes(last_chat)["gen_ai.input.messages"]) == [
        {"role": "user", "parts": [{"type": "text", "content": "Add pack"}]},
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "id": "call_0", "name": "add_todo", "arguments": {"text": "pack"}}],
        },
        {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call_0", "response": "added"}]},
    ]
    assert json.loads(attributes(last_chat)["gen_ai.output.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Done"}], "finish_reason": "end"}
    ]
    call = attributes(tracing.span("execute_tool add_todo"))
    assert (json.loads(call["gen_ai.tool.call.arguments"]), call["gen_ai.tool.call.result"]) == (
        {"text": "pack"},
        "added",
    )


async def test_the_standard_environment_variable_turns_content_on(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "span_only")

    await run_with_content(fake_openai, make_agent, tracing.instrumentation)

    assert content_attributes(tracing) == CONTENT_ATTRIBUTES


# Metrics ------------------------------------------------------------------------------------------


async def test_model_requests_record_the_client_metrics(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    fake_openai.queue(tool_calls_response(tool_call("get_weather", city="Vigo")), text_response("Sunny"))

    await make_agent(tools=[get_weather], instrument=tracing.instrumentation).run("Weather?")

    data = tracing.metrics.get_metrics_data()
    assert data is not None
    histograms = {
        metric.name: [point for point in metric.data.data_points if isinstance(point, HistogramDataPoint)]
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert set(histograms) == {
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.duration",
        "gen_ai.client.operation.time_to_first_chunk",
    }
    tokens = {
        (point.attributes or {})["gen_ai.token.type"]: point.sum for point in histograms["gen_ai.client.token.usage"]
    }
    assert tokens == {"input": 20, "output": 10}
    [duration] = histograms["gen_ai.client.operation.duration"]
    assert duration.count == 2
    assert dict(duration.attributes or {}) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "test-model",
        "server.address": "127.0.0.1",
        "server.port": urlsplit(fake_openai.base_url).port,
    }


# Not traced ---------------------------------------------------------------------------------------


async def test_runs_are_not_traced_unless_asked(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    assert recorder_for(None) is None  # `instrument()` was not called
    assert recorder_for(False) is None
    # Asked, but the app set up no SDK: the API's stand-in provider would drop every span.
    assert recorder_for(True) is None
    assert recorder_for(Instrumentation()) is None

    fake_openai.queue(text_response("Hi"))
    assert (await make_agent().run("Hi")).text == "Hi"


async def test_instrument_traces_every_agent_except_those_that_say_no(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pyox_agents.telemetry, "_instrumentation", None)  # undone after the test
    instrument(tracer_provider=tracing.instrumentation.tracer_provider)
    fake_openai.queue(text_response("Hi"), text_response("Hi"))

    await make_agent(name="traced").run("Hi")
    await make_agent(name="quiet", instrument=False).run("Hi")

    assert span_tree(tracing.spans) == "invoke_agent traced\n  chat test-model"


async def test_a_run_event_stream_is_the_same_traced_or_not(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    def describe(events: list[Any]) -> list[str]:
        return [type(event).__name__ for event in events]

    fake_openai.queue(*[tool_calls_response(tool_call("get_weather", city="Vigo")), text_response("Sunny")] * 2)
    plain = await collect_events(make_agent(tools=[get_weather]).execute("Weather?"))
    traced = await collect_events(
        make_agent(tools=[get_weather], instrument=tracing.instrumentation).execute("Weather?")
    )

    assert describe(plain) == describe(traced)
    assert run_finished(plain).text == run_finished(traced).text


@pytest.mark.parametrize(
    ("variable", "argument", "captured"),
    [
        ("SPAN_ONLY", None, True),
        ("span_and_event", None, True),
        ("EVENT_ONLY", None, True),
        ("NO_CONTENT", None, False),
        ("", None, False),
        ("SPAN_ONLY", False, False),  # the argument wins
        ("", True, True),
    ],
)
def test_the_content_switch(
    monkeypatch: pytest.MonkeyPatch, variable: str, argument: bool | None, captured: bool
) -> None:
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", variable)

    assert pyox_agents.telemetry._capture_content(argument) is captured


def test_images_and_files_are_named_in_content_but_never_copied() -> None:
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "What is this?"}, {"type": "image", "url": "data:..."}],
    }

    assert pyox_agents.telemetry._semconv_message(message) == {
        "role": "user",
        "parts": [{"type": "text", "content": "What is this?"}, {"type": "image"}],
    }
