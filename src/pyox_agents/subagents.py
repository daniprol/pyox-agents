"""Subagents: an agent the model calls like a tool.

    researcher = Agent(model="gpt-5-mini", prompt="Research topics. Be thorough.", tools=[web_search])
    writer = Agent(
        model="gpt-5",
        prompt="Write articles. Use `research` for facts.",
        tools=[researcher.as_tool("research", "Research a topic and report what you found.")],
    )

The model calls `research(task=...)`. The researcher runs with a fresh
conversation (the task is its first message), and its final answer is the
tool's result. Its events come in the same stream, between `SubagentStarted`
and `SubagentFinished`, with `event.subagent_id` set to the id of the tool
call running it. Its approvals and `ctx.ask` questions go to the person, like
the main agent's, with ids like `"call_0/call_1"`. Its tools share the run's
`deps`, and its tokens are counted in the run's usage.

A subagent cannot have subagents of its own.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pyox_agents.agent import Agent

TalkTo = Literal["user", "parent"]
"""Who a subagent's model can ask questions, besides doing its task:

- `None`: nobody (default). It does the task and answers.
- `"user"`: the person, with an `ask_user(question)` tool. The question waits like a `ctx.ask`.
- `"parent"`: the calling agent, with an `ask_parent(question)` tool. The subagent stops, and the
  calling agent reads the question as the tool's result, with a `subagent_id`. It answers, or
  follows up later, by calling the tool again with that `subagent_id`: the subagent goes on with
  its conversation.

Approvals and `ctx.ask` questions from the subagent's tools always go to the person.
"""

StateAccess = Literal["read", "write"]
"""What a subagent may do with the run's state (`ctx.state`): `None` (nothing, default), `"read"` or `"write"`."""


@dataclass(frozen=True, slots=True)
class Subagent:
    """A tool that runs `agent`. Make it with `agent.as_tool(...)`."""

    agent: "Agent"
    name: str
    description: str
    inherit_tools: bool = False  # also give it the calling agent's tools (not its subagents)
    talk_to: TalkTo | None = None
    state: StateAccess | None = None
