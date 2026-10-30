"""Human in the loop: tool approvals, questions from tools, and their replies.

A tool call can wait for a person in three ways:

- **Approval**: a tool made with `@tool(requires_approval=True)` (or a hook
  raising `ApprovalRequired`) only runs once someone approves it.
- **A question**: a tool calls `answer = await ctx.ask(data, metadata=...)`
  anywhere in its body.
- **A frontend tool** (`@tool(frontend=True)` or `Tool.frontend_tool(...)`): the app
  runs it and sends back its result.

Each one becomes an `InputRequested`, answered with `Answer`, `Approve` or
`Deny`, in one of two ways:

- **Live**: `agent.run(..., on_request=handler)`. The handler replies while
  the run waits, and the tool goes on from where it asked.
- **Stop and resume**: without a handler (or when it returns `None`), the run
  ends with `result.pending`. A later run passes `answers={call_id: reply}`
  and continues from the history alone. The waiting tool runs again from its
  start: earlier `ctx.ask` calls return their stored answers at once, and
  `ctx.step(...)` returns saved results, so put side effects after the
  question or inside a step.

`data` may reach the model (for example, if the tool returns it); `metadata`
never does: it is for your app and your frontend only.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pyox_agents._core import InputRequested


@dataclass(frozen=True, slots=True)
class Answer:
    """The reply to a question (`ctx.ask`) or the result of a frontend tool."""

    value: Any = None
    metadata: Any = None  # for the tool, never sent to the model

    def _to_json(self) -> dict[str, Any]:
        return {"type": "answer", "value": self.value, "metadata": self.metadata}


@dataclass(frozen=True, slots=True)
class Approve:
    """Let the tool run. `arguments` replaces the arguments the model chose."""

    arguments: dict[str, Any] | None = None
    metadata: Any = None  # for the tool (`ctx.approval.metadata`), never sent to the model

    def _to_json(self) -> dict[str, Any]:
        return {"type": "approve", "arguments": self.arguments, "metadata": self.metadata}


@dataclass(frozen=True, slots=True)
class Deny:
    """Refuse: the tool does not run (or stops at its question). The model reads `message` as an error result."""

    message: str = "The user denied this tool call"
    metadata: Any = None

    def _to_json(self) -> dict[str, Any]:
        return {"type": "deny", "message": self.message, "metadata": self.metadata}


Reply = Answer | Approve | Deny
"""What a person (or the app) sends back for an `InputRequested`."""

OnRequest = Callable[[InputRequested], Reply | Awaitable[Reply | None] | None]
"""A live handler: `agent.run(..., on_request=handler)`. Return `None` to stop the run instead."""


class ApprovalRequired(Exception):  # noqa: N818 (it asks for something; it is not an error)
    """Raise it in a `before_tool_call` hook to make that call wait for approval.

        def guard(ctx: RunContext, call: ToolCall) -> None:
            if "rm " in call["arguments"].get("command", ""):
                raise ApprovalRequired({"command": call["arguments"]["command"]}, metadata={"risk": "high"})

    Once the call is approved, the hook may raise it again (the run replays the
    call): the stored approval is used.
    """

    def __init__(self, data: Any = None, *, metadata: Any = None) -> None:
        super().__init__("This tool call needs approval")
        self.data = data
        self.metadata = metadata
