"""pyox-agents: minimal and fast AI agents for Python, with a Rust core."""

from pyox_agents._core import (
    AgentError,
    ApiError,
    AuthenticationError,
    BuiltinTool,
    Client,
    HistoryError,
    InvalidResponseError,
    MaxTurnsExceededError,
    NetworkError,
    RateLimitError,
    RequestTimeoutError,
    Run,
    RunFinished,
    TextDelta,
    ToolFinished,
    ToolStarted,
)
from pyox_agents.agent import Agent, TextChunks
from pyox_agents.messages import (
    AssistantMessage,
    History,
    HistoryStore,
    LoadHistory,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from pyox_agents.tools import Tool, ToolLike, tool

AgentEvent = TextDelta | ToolStarted | ToolFinished | RunFinished
"""Any event yielded while iterating over a run."""

__all__ = [
    "Agent",
    "AgentError",
    "AgentEvent",
    "ApiError",
    "AssistantMessage",
    "AuthenticationError",
    "BuiltinTool",
    "Client",
    "History",
    "HistoryError",
    "HistoryStore",
    "InvalidResponseError",
    "LoadHistory",
    "MaxTurnsExceededError",
    "Message",
    "NetworkError",
    "RateLimitError",
    "RequestTimeoutError",
    "Run",
    "RunFinished",
    "TextChunks",
    "TextDelta",
    "Tool",
    "ToolCall",
    "ToolFinished",
    "ToolLike",
    "ToolMessage",
    "ToolStarted",
    "UserMessage",
    "tool",
]
