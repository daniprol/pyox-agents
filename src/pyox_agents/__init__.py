"""pyox-agents: minimal and fast AI agents for Python, with a Rust core."""

from importlib.metadata import version as _version

from pyox_agents._core import (
    AgentError,
    ApiError,
    AuthenticationError,
    BuiltinTool,
    ChoiceAnswer,
    Client,
    Decision,
    EventBatches,
    HistoryError,
    HookError,
    InputRequested,
    InvalidResponseError,
    MaxTurnsExceededError,
    NetworkError,
    PendingRequestsError,
    PredicateAnswer,
    ProtocolStream,
    RateLimitError,
    Refusal,
    RequestTimeoutError,
    Run,
    RunContext,
    RunFinished,
    ScoreAnswer,
    StateChanged,
    SubagentFinished,
    SubagentStarted,
    TextDelta,
    ToolFinished,
    ToolStarted,
    Usage,
)
from pyox_agents.agent import Agent, Api, TextChunks, Unanswered
from pyox_agents.content import file, image
from pyox_agents.decisions import Choice, Predicate, Question, Score, decide
from pyox_agents.hooks import Hooks, ToolResult
from pyox_agents.human_input import Answer, ApprovalRequired, Approve, Deny, OnRequest, Reply
from pyox_agents.messages import (
    AssistantMessage,
    ContentPart,
    FilePart,
    History,
    HistoryStore,
    ImagePart,
    LoadHistory,
    Message,
    Prompt,
    Reasoning,
    Request,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from pyox_agents.subagents import StateAccess, Subagent, TalkTo
from pyox_agents.tools import RequiresApproval, Tool, ToolLike, ToolReturn, tool

DecisionAnswer = PredicateAnswer | ChoiceAnswer | ScoreAnswer | Refusal
"""Any answer in a `Decision`."""

AgentEvent = (
    TextDelta
    | ToolStarted
    | ToolFinished
    | InputRequested
    | StateChanged
    | SubagentStarted
    | SubagentFinished
    | RunFinished
)
"""Any event yielded while iterating over a run."""

__version__ = _version("pyox-agents")

__all__ = [
    "Agent",
    "AgentError",
    "AgentEvent",
    "Answer",
    "Api",
    "ApiError",
    "ApprovalRequired",
    "Approve",
    "AssistantMessage",
    "AuthenticationError",
    "BuiltinTool",
    "Choice",
    "ChoiceAnswer",
    "Client",
    "ContentPart",
    "Decision",
    "DecisionAnswer",
    "Deny",
    "EventBatches",
    "FilePart",
    "History",
    "HistoryError",
    "HistoryStore",
    "HookError",
    "Hooks",
    "ImagePart",
    "InputRequested",
    "InvalidResponseError",
    "LoadHistory",
    "MaxTurnsExceededError",
    "Message",
    "NetworkError",
    "OnRequest",
    "PendingRequestsError",
    "Predicate",
    "PredicateAnswer",
    "Prompt",
    "ProtocolStream",
    "Question",
    "RateLimitError",
    "Reasoning",
    "Refusal",
    "Reply",
    "Request",
    "RequestTimeoutError",
    "RequiresApproval",
    "Run",
    "RunContext",
    "RunFinished",
    "Score",
    "ScoreAnswer",
    "StateAccess",
    "StateChanged",
    "Subagent",
    "SubagentFinished",
    "SubagentStarted",
    "TalkTo",
    "TextChunks",
    "TextDelta",
    "TextPart",
    "Tool",
    "ToolCall",
    "ToolFinished",
    "ToolLike",
    "ToolMessage",
    "ToolResult",
    "ToolReturn",
    "ToolStarted",
    "Unanswered",
    "Usage",
    "UserMessage",
    "decide",
    "file",
    "image",
    "tool",
]
