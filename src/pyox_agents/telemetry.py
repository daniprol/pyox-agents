"""Tracing runs with OpenTelemetry: `pyox_agents.instrument()` (see docs/telemetry.md).

    pip install "pyox-agents[otel]"

    pyox_agents.instrument()  # after your app set up its TracerProvider (Logfire, an OTLP exporter...)

Each run becomes an `invoke_agent` span, with a `chat` span for each model request and an
`execute_tool` span for each tool call (a subagent's `invoke_agent` goes under its tool call).
Your own spans inside tools and hooks nest under them. pyox-agents only uses the OpenTelemetry
API: it never sets up providers or exporters.

The names follow the OpenTelemetry GenAI semantic conventions, which are still in development:
they may change in a minor release (always listed in the changelog).

How it works: the agent loop (Rust, `src/telemetry.rs`) calls the private `_Recorder` below at
fixed points, with times it measured itself. This file is the only place that knows OpenTelemetry.
"""

import contextvars
import functools
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

try:
    from opentelemetry import context as otel_context
    from opentelemetry import metrics, trace
    from opentelemetry.trace import SpanKind, StatusCode

    _INSTALLED = True
except ImportError:  # no `otel` extra: runs are never traced
    _INSTALLED = False

if TYPE_CHECKING:
    from opentelemetry.metrics import MeterProvider
    from opentelemetry.trace import Span, TracerProvider

# Names ----------------------------------------------------------------------------------------
# OpenTelemetry GenAI semantic conventions, and our own `pyox.*` attributes.

OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
SYSTEM = "gen_ai.system"  # the older name of `gen_ai.provider.name`, still read by some backends
AGENT_NAME = "gen_ai.agent.name"
CONVERSATION_ID = "gen_ai.conversation.id"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_ID = "gen_ai.response.id"
FINISH_REASONS = "gen_ai.response.finish_reasons"
TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
CACHED_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
REASONING_TOKENS = "gen_ai.usage.reasoning.output_tokens"
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
TOOL_DEFINITIONS = "gen_ai.tool.definitions"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_TYPE = "gen_ai.tool.type"
TOOL_DESCRIPTION = "gen_ai.tool.description"
TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_RESULT = "gen_ai.tool.call.result"
TOKEN_TYPE = "gen_ai.token.type"
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"
ERROR_TYPE = "error.type"

# `model_parameters` keys, and the attribute each one is recorded as.
REQUEST_PARAMETERS = {
    "temperature": "gen_ai.request.temperature",
    "top_p": "gen_ai.request.top_p",
    "max_tokens": "gen_ai.request.max_tokens",
    "max_completion_tokens": "gen_ai.request.max_tokens",
    "max_output_tokens": "gen_ai.request.max_tokens",
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "seed": "gen_ai.request.seed",
    "stop": "gen_ai.request.stop_sequences",
}

SUBAGENT_ID = "pyox.subagent_id"
RUN_TURNS = "pyox.run.turns"
RUN_STOP_REASON = "pyox.run.stop_reason"
RUN_PENDING = "pyox.run.pending"
# A run's total tokens (subagents included). Not `gen_ai.usage.*`, which only `chat` spans have, so
# backends that add up `gen_ai.usage.*` over a trace never count a token twice.
RUN_USAGE = {
    "input_tokens": "pyox.usage.input_tokens",
    "output_tokens": "pyox.usage.output_tokens",
    "cached_input_tokens": "pyox.usage.cached_input_tokens",
    "reasoning_tokens": "pyox.usage.reasoning_tokens",
}
TURN = "pyox.turn"
RETRIES = "pyox.http.retries"
OPENAI_REQUEST_ID = "pyox.openai.request_id"
TOOL_KIND = "pyox.tool.kind"
TOOL_WAITING = "pyox.tool.waiting"

# Events on `execute_tool` spans, and their attributes, by the names Rust uses.
EVENTS = {
    "input_requested": "pyox.input_requested",
    "input_replied": "pyox.input_replied",
    "state_changed": "pyox.state_changed",
}
EVENT_ATTRIBUTES = {
    "kind": "pyox.input.kind",
    "type": "pyox.reply.type",
    "source": "pyox.reply.source",
    "paths": "pyox.state.paths",
    "delta": "pyox.state.delta",
}

CAPTURE_CONTENT_VARIABLE = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"

# Histogram buckets the semantic conventions advise.
TOKEN_BUCKETS = [4**power for power in range(14)]  # 1, 4, 16 ... 67108864
DURATION_BUCKETS = [0.01 * 2**power for power in range(14)]  # 0.01 ... 81.92 s
FIRST_CHUNK_BUCKETS = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]

_VERSION = version("pyox-agents")

# Public API -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Instrumentation:
    """How runs are traced: `pyox_agents.instrument(...)`, or `Agent(instrument=Instrumentation(...))`.

    - `tracer_provider`, `meter_provider`: where spans and metrics go. `None` uses the global
      providers (`opentelemetry.trace.set_tracer_provider(...)`, which Logfire and others set up),
      looked up at each run.
    - `capture_content`: also record prompts, answers, tool arguments and results, and state
      changes. They may hold personal data, so it is off by default. `None` reads the standard
      `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` environment variable (`SPAN_ONLY`,
      `SPAN_AND_EVENT` or `EVENT_ONLY` turn it on; pyox-agents records content on spans only).
    """

    tracer_provider: "TracerProvider | None" = None
    meter_provider: "MeterProvider | None" = None
    capture_content: bool | None = None


_instrumentation: Instrumentation | None = None  # set by `instrument()`


def instrument(
    *,
    tracer_provider: "TracerProvider | None" = None,
    meter_provider: "MeterProvider | None" = None,
    capture_content: bool | None = None,
) -> None:
    """Trace the runs of every agent (except those made with `Agent(instrument=False)` or their own
    `Instrumentation`). See `Instrumentation` for the arguments.

    Needs `pip install "pyox-agents[otel]"`. Spans go to the providers your app set up:
    pyox-agents never configures exporters.
    """
    require_opentelemetry()
    global _instrumentation
    _instrumentation = Instrumentation(tracer_provider, meter_provider, capture_content)


def require_opentelemetry() -> None:
    if not _INSTALLED:
        raise ImportError('Tracing needs OpenTelemetry: pip install "pyox-agents[otel]"')


def recorder_for(setting: Instrumentation | bool | None) -> "_Recorder | None":
    """The recorder for one run, or `None` when the run is not traced. Called at `execute()`.

    `setting` is the agent's `instrument=`: `None` follows `instrument()`, `True` traces with the
    defaults, `False` never traces.
    """
    if setting is None:
        setting = _instrumentation
    elif setting is True:
        setting = Instrumentation()
    if not isinstance(setting, Instrumentation):
        return None
    provider = setting.tracer_provider or trace.get_tracer_provider()
    # The API's stand-ins until an SDK is set up: spans would go nowhere, so skip all the work.
    if isinstance(provider, trace.NoOpTracerProvider | trace.ProxyTracerProvider):
        return None
    return _Recorder(setting, provider)


# What Rust calls ------------------------------------------------------------------------------


class _Recorder:
    """Makes the spans of one run and its subagents. Rust calls `start_run` (see `src/telemetry.rs`)."""

    def __init__(self, settings: Instrumentation, provider: "TracerProvider") -> None:
        self.tracer = provider.get_tracer("pyox_agents", _VERSION)
        self.instruments = _instruments(settings.meter_provider or metrics.get_meter_provider())
        self.capture_content = _capture_content(settings.capture_content)
        # The run's parent: what is current when `execute()` is called, even if the run is read
        # later or in another task (a web stream).
        self.parent = otel_context.get_current()

    def start_run(self, *, run_context: contextvars.Context, **fields: Any) -> tuple["_RunSpan", contextvars.Context]:
        run = _RunSpan(self, self.parent, subagent_id=None, **fields)
        return run, _context_with_span(run_context, run.span)


class _RunSpan:
    """An `invoke_agent` span, open while the run works."""

    def __init__(
        self,
        recorder: _Recorder,
        parent: "otel_context.Context",
        *,
        start_ns: int,
        name: str | None,
        model: str,
        base_url: str,
        conversation_id: str | None,
        subagent_id: str | None,
    ) -> None:
        self.recorder = recorder
        self.name = name
        self.server = _server_attributes(base_url)
        attributes = {
            OPERATION: "invoke_agent",
            PROVIDER: "openai",
            SYSTEM: "openai",
            AGENT_NAME: name,
            REQUEST_MODEL: model,
            CONVERSATION_ID: conversation_id,
            SUBAGENT_ID: subagent_id,
        }
        self.span = recorder.tracer.start_span(
            f"invoke_agent {name}" if name else "invoke_agent",
            context=parent,
            kind=SpanKind.INTERNAL,
            start_time=start_ns,
            attributes=_without_none(attributes),
        )
        # Rust only sends content when this is true: an unsampled run costs nothing more.
        self.capture_content = recorder.capture_content and self.span.is_recording()

    def model_request(
        self,
        *,
        turn: int,
        start_ns: int,
        end_ns: int,
        model: str,
        model_parameters: dict[str, Any],
        error: BaseException | None,
        stop_reason: str | None = None,
        first_chunk_ns: int | None = None,
        usage: dict[str, int] | None = None,
        response_id: str | None = None,
        request_id: str | None = None,
        retries: int = 0,
        instructions: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        """One `chat` span, sent after the request: it has no live children."""
        common = {OPERATION: "chat", PROVIDER: "openai", REQUEST_MODEL: model, **self.server}
        attributes: dict[str, Any] = {
            **common,
            SYSTEM: "openai",
            TURN: turn,
            FINISH_REASONS: [stop_reason] if stop_reason else None,
            RESPONSE_ID: response_id,
            OPENAI_REQUEST_ID: request_id or getattr(error, "request_id", None),
            RETRIES: retries,
        }
        for key, value in model_parameters.items():
            if key in REQUEST_PARAMETERS and value is not None:
                attributes[REQUEST_PARAMETERS[key]] = [value] if isinstance(value, str) else value
        if usage is not None:
            attributes[INPUT_TOKENS] = usage["input_tokens"]
            attributes[OUTPUT_TOKENS] = usage["output_tokens"]
            attributes[CACHED_INPUT_TOKENS] = usage["cached_input_tokens"]
            attributes[REASONING_TOKENS] = usage["reasoning_tokens"]
        if first_chunk_ns is not None:
            attributes[TIME_TO_FIRST_CHUNK] = (first_chunk_ns - start_ns) / 1e9
        if messages is not None:  # content capture is on
            if instructions:
                attributes[SYSTEM_INSTRUCTIONS] = json.dumps([{"type": "text", "content": instructions}])
            attributes[INPUT_MESSAGES] = json.dumps([_semconv_message(message) for message in messages])
            attributes[TOOL_DEFINITIONS] = json.dumps([{"type": "function", **tool} for tool in tools or ()])
            if output is not None:
                output_message = {"role": "assistant", "content": output["text"], "tool_calls": output["tool_calls"]}
                attributes[OUTPUT_MESSAGES] = json.dumps(
                    [{**_semconv_message(output_message), "finish_reason": stop_reason}]
                )

        span = self.recorder.tracer.start_span(
            f"chat {model}",
            context=trace.set_span_in_context(self.span),
            kind=SpanKind.CLIENT,
            start_time=start_ns,
            attributes=_without_none(attributes),
        )
        if error is not None:
            _set_error(span, type(error).__name__, str(error))
            common[ERROR_TYPE] = type(error).__name__
        span.end(end_time=end_ns)

        token_usage, duration, time_to_first_chunk = self.recorder.instruments
        duration.record((end_ns - start_ns) / 1e9, common)
        if first_chunk_ns is not None:
            time_to_first_chunk.record((first_chunk_ns - start_ns) / 1e9, common)
        if usage is not None:
            token_usage.record(usage["input_tokens"], {**common, TOKEN_TYPE: "input"})
            token_usage.record(usage["output_tokens"], {**common, TOKEN_TYPE: "output"})

    def start_tool(
        self,
        *,
        run_context: contextvars.Context,
        start_ns: int,
        call_id: str,
        name: str,
        description: str,
        kind: str,
        arguments: Any = None,
    ) -> tuple["_ToolSpan", contextvars.Context]:
        tool = _ToolSpan(self, start_ns, call_id, name, description, kind, arguments)
        return tool, _context_with_span(run_context, tool.span)

    def finish(
        self,
        *,
        end_ns: int,
        usage: dict[str, int],
        error: BaseException | None,
        turns: int | None = None,
        stop_reason: str | None = None,
        pending: int = 0,
    ) -> None:
        attributes: dict[str, Any] = {RUN_USAGE[key]: value for key, value in usage.items() if key in RUN_USAGE}
        attributes[RUN_TURNS] = turns
        attributes[RUN_STOP_REASON] = stop_reason
        attributes[RUN_PENDING] = pending or None
        self.span.set_attributes(_without_none(attributes))
        if error is not None:
            _set_error(self.span, type(error).__name__, str(error))
        self.span.end(end_time=end_ns)

    def cancel(self, end_ns: int) -> None:
        _cancel(self.span, end_ns)


class _ToolSpan:
    """An `execute_tool` span, open while the tool call runs (hooks and approval included)."""

    def __init__(
        self, run: _RunSpan, start_ns: int, call_id: str, name: str, description: str, kind: str, arguments: Any
    ) -> None:
        self.run = run
        self.call_id = call_id
        self.name = name
        attributes = {
            OPERATION: "execute_tool",
            TOOL_NAME: name,
            TOOL_CALL_ID: call_id,
            TOOL_TYPE: "function",
            TOOL_DESCRIPTION: description or None,
            AGENT_NAME: run.name,
            TOOL_KIND: kind,
            TOOL_ARGUMENTS: None if arguments is None else json.dumps(arguments),
        }
        self.span = run.recorder.tracer.start_span(
            f"execute_tool {name}",
            context=trace.set_span_in_context(run.span),
            kind=SpanKind.INTERNAL,
            start_time=start_ns,
            attributes=_without_none(attributes),
        )
        self.capture_content = run.capture_content

    def start_run(
        self, *, run_context: contextvars.Context, name: str | None, **fields: Any
    ) -> tuple[_RunSpan, contextvars.Context]:
        """A subagent's run, under this call. Without a name, it is named after the tool."""
        parent = trace.set_span_in_context(self.span)
        run = _RunSpan(self.run.recorder, parent, name=name or self.name, subagent_id=self.call_id, **fields)
        return run, _context_with_span(run_context, run.span)

    def add_event(self, name: str, attributes: dict[str, Any], time_ns: int) -> None:
        if "delta" in attributes:  # a state change: always its paths, the patch itself only with content
            delta = attributes.pop("delta")
            attributes["paths"] = [operation["path"] for operation in delta]
            if self.capture_content:
                attributes["delta"] = json.dumps(delta)
        named = {EVENT_ATTRIBUTES[key]: value for key, value in attributes.items()}
        self.span.add_event(EVENTS[name], named, timestamp=time_ns)

    def finish(
        self,
        *,
        end_ns: int,
        error: BaseException | None,
        is_error: bool = False,
        waiting: bool = False,
        result: str | None = None,
    ) -> None:
        if waiting:  # stopped for a person: not an error
            self.span.set_attribute(TOOL_WAITING, True)
        if result is not None:
            self.span.set_attribute(TOOL_RESULT, result)
        if error is not None:
            _set_error(self.span, type(error).__name__, str(error))
        elif is_error:  # the model got an error result; the run goes on
            _set_error(self.span, "tool_error", result)
        self.span.end(end_time=end_ns)

    def cancel(self, end_ns: int) -> None:
        _cancel(self.span, end_ns)


# Helpers --------------------------------------------------------------------------------------


def _context_with_span(run_context: contextvars.Context, span: "Span") -> contextvars.Context:
    """A copy of the run's contextvars where `span` is the current span.

    A copy for each span: parallel tool calls each get their own, and a `contextvars.Context`
    cannot be entered by two threads at once anyway.
    """
    context = run_context.copy()
    context.run(lambda: otel_context.attach(trace.set_span_in_context(span)))
    return context


def _set_error(span: "Span", error_type: str, message: str | None) -> None:
    span.set_attribute(ERROR_TYPE, error_type)
    span.set_status(StatusCode.ERROR, message)


def _cancel(span: "Span", end_ns: int) -> None:
    """The run was dropped before this span ended (Python stopped reading it)."""
    _set_error(span, "cancelled", "The run was dropped before it finished")
    span.end(end_time=end_ns)


def _without_none(attributes: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in attributes.items() if value is not None}


def _capture_content(setting: bool | None) -> bool:
    if setting is not None:
        return setting
    value = os.environ.get(CAPTURE_CONTENT_VARIABLE, "").strip().lower()
    return value in {"span_only", "span_and_event", "event_only", "true"}


def _server_attributes(base_url: str) -> dict[str, Any]:
    url = urlsplit(base_url)
    return _without_none(
        {SERVER_ADDRESS: url.hostname, SERVER_PORT: url.port or (443 if url.scheme == "https" else 80)}
    )


@functools.cache  # one set of instruments per meter provider
def _instruments(meter_provider: "MeterProvider") -> tuple[Any, Any, Any]:
    meter = meter_provider.get_meter("pyox_agents", _VERSION)
    token_usage = meter.create_histogram(
        "gen_ai.client.token.usage",
        unit="{token}",
        description="Tokens used per model request",
        explicit_bucket_boundaries_advisory=TOKEN_BUCKETS,
    )
    duration = meter.create_histogram(
        "gen_ai.client.operation.duration",
        unit="s",
        description="Duration of model requests",
        explicit_bucket_boundaries_advisory=DURATION_BUCKETS,
    )
    time_to_first_chunk = meter.create_histogram(
        "gen_ai.client.operation.time_to_first_chunk",
        unit="s",
        description="Time from sending a model request to its first streamed chunk",
        explicit_bucket_boundaries_advisory=FIRST_CHUNK_BUCKETS,
    )
    return token_usage, duration, time_to_first_chunk


def _semconv_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """A history message (see `pyox_agents.messages`) in the semantic conventions' message format."""
    role = message["role"]
    if role == "tool":
        part = {"type": "tool_call_response", "id": message["tool_call_id"], "response": _text(message["content"])}
        return {"role": "tool", "parts": [part]}
    parts: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str):
        if content:
            parts.append({"type": "text", "content": content})
    else:  # text, images and files: images and files are named, never copied
        parts.extend(_semconv_part(part) for part in content or ())
    for call in message.get("tool_calls", ()):
        parts.append({"type": "tool_call", "id": call["id"], "name": call["name"], "arguments": call["arguments"]})
    return {"role": role, "parts": parts}


def _semconv_part(part: Mapping[str, Any]) -> dict[str, Any]:
    if part["type"] == "text":
        return {"type": "text", "content": part["text"]}
    return {"type": part["type"]}


def _text(content: str | list[Mapping[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part["type"] == "text")
