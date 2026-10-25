"""pyox-agents: minimal and fast AI agents for Python, with a Rust core."""

from pyox_agents._core import (
    AgentError,
    ApiError,
    AuthenticationError,
    BuiltinTool,
    Client,
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
from pyox_agents.agent import Agent
from pyox_agents.tools import Tool, tool

AgentEvent = TextDelta | ToolStarted | ToolFinished | RunFinished
"""Any event yielded while iterating over a run."""

__all__ = [
    "Agent",
    "AgentError",
    "AgentEvent",
    "ApiError",
    "AuthenticationError",
    "BuiltinTool",
    "Client",
    "InvalidResponseError",
    "MaxTurnsExceededError",
    "NetworkError",
    "RateLimitError",
    "RequestTimeoutError",
    "Run",
    "RunFinished",
    "TextDelta",
    "Tool",
    "ToolFinished",
    "ToolStarted",
    "tool",
]
