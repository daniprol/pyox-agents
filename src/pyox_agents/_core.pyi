"""Type stubs for the compiled Rust module (`src/*.rs`)."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Generic, Literal, Self, TypeAlias, final

from typing_extensions import TypeVar

from pyox_agents.human_input import Answer, Approve
from pyox_agents.messages import ContentPart, History, HistoryStore, LoadHistory, Message, Prompt, StopReason

@final
class Client:
    """Connection settings and shared connection pools (HTTP, and WebSocket with `transport="websocket"`).

    `transport="websocket"` sends Responses API requests over WebSocket: a run keeps one connection for all its
    turns, so each turn after the first sends only the new tool results. Chat Completions requests always use HTTP.
    """

    def __new__(
        cls,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        connect_timeout_seconds: float = 10.0,
        request_timeout_seconds: float | None = 600.0,
        max_retries: int = 2,
        transport: Literal["http", "websocket"] = "http",
    ) -> Self: ...
    @staticmethod
    def azure(
        endpoint: str | None = None,
        api_key: str | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        connect_timeout_seconds: float = 10.0,
        request_timeout_seconds: float | None = 600.0,
        max_retries: int = 2,
        transport: Literal["http", "websocket"] = "http",
    ) -> Client:
        """Azure OpenAI (Microsoft Foundry) v1 API; `model` is then your deployment name."""
    @property
    def base_url(self) -> str: ...

@final
class Usage:
    """Token counts reported by the API; counts it did not report are 0."""

    def __new__(
        cls,
        *,
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        requests: int = 0,
    ) -> Self: ...
    @property
    def input_tokens(self) -> int: ...
    @property
    def cached_input_tokens(self) -> int:
        """The part of `input_tokens` read from the prompt cache."""
    @property
    def output_tokens(self) -> int: ...
    @property
    def reasoning_tokens(self) -> int:
        """The part of `output_tokens` the model spent reasoning."""
    @property
    def requests(self) -> int:
        """Model requests that reported usage."""
    def __add__(self, other: Usage, /) -> Usage: ...

DepsT = TypeVar("DepsT", default=Any)

@final
class RunContext(Generic[DepsT]):
    """The `ctx` given to tools that ask for it and to hooks."""

    @property
    def deps(self) -> DepsT: ...
    @property
    def conversation_id(self) -> str | None: ...
    @property
    def turn(self) -> int: ...
    @property
    def usage(self) -> Usage:
        """Tokens used by the run so far."""
    @property
    def tool_call_id(self) -> str | None: ...
    @property
    def tool_name(self) -> str | None: ...
    @property
    def approval(self) -> Approve | None:
        """Inside a tool that needed approval: the `Approve` reply. `None` otherwise."""
    def ask(self, data: Any = None, *, metadata: Any = None, key: str | None = None) -> Awaitable[Answer]:
        """Wait for a person's reply. See `pyox_agents.human_input`."""
    def step(self, name: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Awaitable[Any]:
        """Run `function` once and save its (JSON) result, reused when the call runs again."""
    def ask_sync(self, data: Any = None, *, metadata: Any = None, key: str | None = None) -> Answer:
        """`ask` for sync tools: wait right here for a person's reply."""
    def step_sync(self, name: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """`step` for sync tools (`function` may be async)."""
    @property
    def state(self) -> Any:
        """The run's state (`execute(state=...)`). Change it in place or assign it, inside a tool."""
    @state.setter
    def state(self, new_state: Any) -> None: ...

@final
class TextDelta:
    @property
    def text(self) -> str: ...
    @property
    def subagent_id(self) -> str | None:
        """The subagent writing it (the id of the tool call running it); `None` for the main agent."""

@final
class ToolStarted:
    @property
    def call_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def arguments(self) -> Any: ...
    @property
    def subagent_id(self) -> str | None: ...

@final
class ToolFinished:
    @property
    def call_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def result(self) -> str:
        """The result's text (without its images and files, if it has any)."""
    @property
    def content(self) -> str | list[ContentPart]:
        """The whole result: text, or parts when it has images or files."""
    @property
    def is_error(self) -> bool: ...
    @property
    def metadata(self) -> Any:
        """From `ToolReturn(metadata=...)`; never sent to the model."""
    @property
    def subagent_id(self) -> str | None: ...

@final
class StateChanged:
    """A tool changed the run's state."""

    @property
    def state(self) -> Any:
        """The whole state, as it is now."""
    @property
    def delta(self) -> list[dict[str, Any]]:
        """What changed, as a JSON Patch (RFC 6902)."""
    @property
    def subagent_id(self) -> str | None: ...

@final
class SubagentStarted:
    """A tool call started a subagent. Its events follow, with `subagent_id` set to `call_id`."""

    @property
    def call_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def task(self) -> str: ...
    @property
    def subagent_id(self) -> str | None: ...

@final
class SubagentFinished:
    """A subagent finished, failed (`is_error`), or waits for a person (`stop_reason == "waiting"`)."""

    @property
    def call_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def result(self) -> str: ...
    @property
    def is_error(self) -> bool: ...
    @property
    def stop_reason(self) -> StopReason | None:
        """Why the subagent's run ended (`"waiting"`: it waits for a person); `None` when it failed."""
    @property
    def usage(self) -> Usage: ...
    @property
    def subagent_id(self) -> str | None: ...

@final
class InputRequested:
    """A tool call waiting for a person: an approval, a `ctx.ask` question or a frontend tool."""

    @property
    def call_id(self) -> str: ...
    @property
    def tool_name(self) -> str: ...
    @property
    def arguments(self) -> Any: ...
    @property
    def kind(self) -> Literal["approval", "input", "frontend"]: ...
    @property
    def key(self) -> str | None: ...
    @property
    def data(self) -> Any: ...
    @property
    def metadata(self) -> Any:
        """For the app only; never sent to the model."""
    @property
    def subagent_id(self) -> str | None:
        """The subagent whose tool asks; `None` for the main agent."""

OutputT = TypeVar("OutputT", default=Any)

@final
class RunFinished(Generic[OutputT]):
    @property
    def text(self) -> str: ...
    @property
    def output(self) -> OutputT | None:
        """The answer as `Agent(output_type=...)`; `None` without one, or when the run did not answer."""
    @property
    def turns(self) -> int: ...
    @property
    def usage(self) -> Usage: ...
    @property
    def response_id(self) -> str | None: ...
    @property
    def stop_reason(self) -> StopReason:
        """Why the run ended: "end", "waiting" (see `pending`), "length", "refusal" or "stopped"."""
    @property
    def pending(self) -> list[InputRequested]:
        """Requests of tool calls waiting for a person; empty when the run finished normally."""
    @property
    def messages(self) -> list[Message]: ...
    @property
    def new_messages(self) -> list[Message]: ...
    @property
    def state(self) -> Any:
        """The state at the end of the run: give it back with `execute(state=...)`."""
    def messages_json(self) -> str: ...
    def new_messages_json(self) -> str: ...

_Event: TypeAlias = (
    TextDelta
    | ToolStarted
    | ToolFinished
    | InputRequested
    | StateChanged
    | SubagentStarted
    | SubagentFinished
    | RunFinished[OutputT]
)

@final
class Run(Generic[OutputT]):
    def __aiter__(self) -> Run[OutputT]: ...
    async def __anext__(self) -> _Event[OutputT]: ...
    def stop(self) -> None:
        """End the run at the next safe point, keeping the partial answer; it ends with
        `RunFinished(stop_reason="stopped")`."""
    def batches(self) -> EventBatches[OutputT]:
        """Iterate over lists of events: each holds every event already waiting (at least one)."""
    def ai_sdk(self) -> ProtocolStream:
        """The run as a Vercel AI SDK UI message stream (server-sent events text)."""
    def ag_ui(self, thread_id: str, run_id: str) -> ProtocolStream:
        """The run as an AG-UI event stream (server-sent events text)."""

@final
class SyncRun(Generic[OutputT]):
    """A run for code without an event loop, from `Agent.execute_sync`: `for event in run`."""

    def __iter__(self) -> SyncRun[OutputT]: ...
    def __next__(self) -> _Event[OutputT]: ...
    def __enter__(self) -> SyncRun[OutputT]: ...
    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        """Close the run."""
    def close(self) -> None:
        """End the run at once and keep nothing, like `break`."""
    def stop(self) -> None:
        """End the run at the next safe point, keeping the partial answer (from any thread)."""
    def batches(self) -> EventBatches[OutputT]:
        """Iterate over lists of events: each holds every event already waiting (at least one)."""
    def ai_sdk(self) -> ProtocolStream:
        """The run as a Vercel AI SDK UI message stream (server-sent events text)."""
    def ag_ui(self, thread_id: str, run_id: str) -> ProtocolStream:
        """The run as an AG-UI event stream (server-sent events text)."""

@final
class PredicateAnswer:
    @property
    def name(self) -> str: ...
    @property
    def probability(self) -> float:
        """How likely the statement is true, from 0 to 1."""

@final
class ChoiceAnswer:
    @property
    def name(self) -> str: ...
    @property
    def choice(self) -> str | bool:
        """The most likely option."""
    @property
    def confidence(self) -> float: ...
    @property
    def probabilities(self) -> dict[str | bool, float]:
        """`{option: probability}`, in the order of the options."""

@final
class ScoreAnswer:
    @property
    def name(self) -> str: ...
    @property
    def score(self) -> float:
        """The probability-weighted average of the level indices (0 is the lowest level)."""
    @property
    def confidence(self) -> float: ...
    @property
    def probabilities(self) -> dict[str, float]:
        """`{level label: probability}`, from the lowest level to the highest."""

@final
class Refusal:
    """The model declined to answer this question. The other answers are still valid."""

    @property
    def name(self) -> str: ...

DecisionAnswer = PredicateAnswer | ChoiceAnswer | ScoreAnswer | Refusal

@final
class Decision:
    """The answers to one `decide(...)` call, by question name: `decision["urgent"]`."""

    @property
    def model(self) -> str: ...
    @property
    def usage(self) -> Usage: ...
    @property
    def request_id(self) -> str | None:
        """OpenAI's id for this request, for support."""
    @property
    def answers(self) -> dict[str, DecisionAnswer]: ...
    def __getitem__(self, name: str, /) -> DecisionAnswer: ...
    def __len__(self) -> int: ...

def send_decision(
    client: Client,
    input: Prompt | Sequence[Message],  # noqa: A002
    questions: list[dict[str, Any]],
    *,
    model: str,
    safety_identifier: str | None = None,
    prompt_cache_key: str | None = None,
) -> Awaitable[Decision]: ...

@final
class EventBatches(Generic[OutputT]):
    """`async for` for a `Run`, `for` for a `SyncRun`."""

    def __aiter__(self) -> EventBatches[OutputT]: ...
    async def __anext__(self) -> list[_Event[OutputT]]: ...
    def __iter__(self) -> EventBatches[OutputT]: ...
    def __next__(self) -> list[_Event[OutputT]]: ...

@final
class ProtocolStream:
    """`async for` for a `Run`, `for` for a `SyncRun`."""

    def __aiter__(self) -> ProtocolStream: ...
    async def __anext__(self) -> str: ...
    def __iter__(self) -> ProtocolStream: ...
    def __next__(self) -> str: ...
    def close(self) -> None:
        """End the run at once (a web server calls it when the client went away)."""

@final
class Engine:
    def __new__(
        cls,
        *,
        client: Client,
        model: str,
        name: str | None = None,
        instructions: str = ...,
        tools: Sequence[Any] = ...,
        model_parameters: Mapping[str, Any] | None = None,
        max_turns: int = 20,
        text_chunks: str | int | None = None,
        history_store: HistoryStore | LoadHistory | None = None,
        hooks: Sequence[object] = ...,
        api: str = "chat",
        store: bool = False,
        encrypted_reasoning: bool = True,
        unanswered: str = "close",
        output_schema: Mapping[str, Any] | None = None,
        output_parser: Callable[[str], Any] | None = None,
    ) -> Self: ...
    def execute(
        self,
        prompt: Prompt | None = None,
        *,
        history: History | None = None,
        conversation_id: str | None = None,
        text_chunks: str | int | None = None,
        deps: Any = None,
        previous_response_id: str | None = None,
        answers: Mapping[str, Any] | None = None,
        on_request: Callable[[InputRequested], Any] | None = None,
        tools: Sequence[Any] | None = None,
        frontend_tools: Sequence[Mapping[str, Any]] | None = None,
        state: Any = None,
        telemetry: object | None = None,
        sync: bool = False,
    ) -> Run | SyncRun: ...

class AgentError(Exception):
    usage: Usage | None
    """Tokens the run used before it failed; `None` if raised outside a run."""
    messages: list[Message] | None
    """The conversation when the run failed (continue with `history=error.messages`); `None` outside a run."""

class ApiError(AgentError):
    status_code: int | None
    """The HTTP status; `None` for an error that came inside a stream."""
    code: str | None
    """OpenAI's error code, like `"rate_limit_exceeded"` or `"context_length_exceeded"`, if it gave one."""
    body: Any
    """The error body: parsed JSON (usually `{"error": {...}}`) when it is JSON, else the text."""
    request_id: str | None
    """OpenAI's id for the request (`x-request-id`), for support."""
    retry_after_seconds: float | None
    """How long the server asked to wait before trying again, if it said."""

class AuthenticationError(ApiError): ...
class RateLimitError(ApiError): ...
class RequestTimeoutError(AgentError): ...
class NetworkError(AgentError): ...
class InvalidResponseError(AgentError): ...
class MaxTurnsExceededError(AgentError): ...

class HistoryError(AgentError):
    """A history store failed. When your store raised, its exception is `__cause__`."""

class HookError(AgentError):
    """A hook (or `on_request`) raised or returned something invalid. When your code raised,
    its exception is `__cause__`."""

class PendingRequestsError(AgentError): ...

class StopRun(AgentError):  # noqa: N818 (an instruction, like `StopIteration`)
    """Raise in a tool or a hook to end the run. The run raises it, with `messages` and `usage`."""

class StopToolCall(BaseException):
    """Raised by `ctx.ask` to stop a tool that must wait or was denied. Do not catch it."""
