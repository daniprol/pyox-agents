"""Type stubs for the compiled Rust module (`src/*.rs`)."""

from collections.abc import Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar, final

from pyox_agents.messages import History, HistoryStore, LoadHistory, Message, Prompt

@final
class Client:
    """Connection settings and shared connection pools (HTTP, and WebSocket with `transport="websocket"`).

    `transport="websocket"` sends Responses API requests over WebSocket: a run keeps one connection for all its
    turns, so each turn after the first sends only the new tool results. Chat Completions requests always use HTTP.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        headers: Mapping[str, str] | None = None,
        connect_timeout_seconds: float = 10.0,
        request_timeout_seconds: float | None = 600.0,
        max_retries: int = 2,
        transport: Literal["http", "websocket"] = "http",
    ) -> None: ...
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
class BuiltinTool:
    """A tool implemented in Rust inside pyox-agents, selected by name."""

    def __init__(self, name: str) -> None: ...
    @staticmethod
    def available() -> list[str]: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...

@final
class Usage:
    """Token counts reported by the API; counts it did not report are 0."""

    def __init__(
        self,
        *,
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
        requests: int = 0,
    ) -> None: ...
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
    def __add__(self, other: Usage) -> Usage: ...

DepsT = TypeVar("DepsT")

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

@final
class TextDelta:
    @property
    def text(self) -> str: ...

@final
class ToolStarted:
    @property
    def call_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def arguments(self) -> Any: ...

@final
class ToolFinished:
    @property
    def call_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def result(self) -> str: ...
    @property
    def is_error(self) -> bool: ...

@final
class RunFinished:
    @property
    def text(self) -> str: ...
    @property
    def turns(self) -> int: ...
    @property
    def usage(self) -> Usage: ...
    @property
    def response_id(self) -> str | None: ...
    @property
    def messages(self) -> list[Message]: ...
    @property
    def new_messages(self) -> list[Message]: ...
    def messages_json(self) -> str: ...
    def new_messages_json(self) -> str: ...

@final
class Run:
    def __aiter__(self) -> Run: ...
    async def __anext__(self) -> TextDelta | ToolStarted | ToolFinished | RunFinished: ...

@final
class Engine:
    def __init__(
        self,
        *,
        client: Client,
        model: str,
        prompt: str = "",
        tools: Sequence[Any] = (),
        model_parameters: Mapping[str, Any] | None = None,
        max_turns: int = 20,
        text_chunks: str | int | None = None,
        history_store: HistoryStore | LoadHistory | None = None,
        hooks: Sequence[object] = (),
        api: str = "chat",
        store: bool = False,
        encrypted_reasoning: bool = True,
    ) -> None: ...
    def execute(
        self,
        prompt: Prompt | None = None,
        *,
        history: History | None = None,
        conversation_id: str | None = None,
        text_chunks: str | int | None = None,
        deps: Any = None,
        previous_response_id: str | None = None,
    ) -> Run: ...

class AgentError(Exception):
    usage: Usage | None
    """Tokens the run used before it failed; `None` if raised outside a run."""

class ApiError(AgentError): ...
class AuthenticationError(ApiError): ...
class RateLimitError(ApiError): ...
class RequestTimeoutError(AgentError): ...
class NetworkError(AgentError): ...
class InvalidResponseError(AgentError): ...
class MaxTurnsExceededError(AgentError): ...
class HistoryError(AgentError): ...
class HookError(AgentError): ...
