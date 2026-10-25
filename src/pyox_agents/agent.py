"""`Agent`: the main entry point."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeGuard

from pyox_agents._core import BuiltinTool, Client, Engine, Run, RunFinished
from pyox_agents.messages import History, HistoryStore, LoadHistory
from pyox_agents.tools import Tool, ToolLike, tool

TextChunks = Literal["delta", "paragraph", "message"] | int
"""How streamed text is grouped into `TextDelta` events:

- `"delta"`: every piece the API sends (default, lowest latency).
- an int N: every N pieces.
- `"paragraph"`: each markdown paragraph; code blocks are never split.
- `"message"`: the whole text of each assistant message.

Bigger chunks mean fewer events, so less work for Python.
"""


class Agent:
    """A model, a system prompt and tools.

    Create it once and reuse it: each `execute()` call has its own history,
    so many runs can happen at the same time.

        agent = Agent(model="gpt-5-mini", prompt="Be brief.", tools=[get_weather])
        async for event in agent.execute("Weather in Vigo?"):
            ...

    With a `history_store`, a run names its conversation and the agent loads
    it before the run and saves the new messages after it:

        agent = Agent(model="gpt-5-mini", history_store=store)
        await agent.run("Hi again", conversation_id="user-42")
    """

    def __init__(
        self,
        *,
        model: str,
        prompt: str = "",
        tools: Sequence[ToolLike | BuiltinTool | Callable[..., Any]] = (),
        client: Client | None = None,
        model_parameters: Mapping[str, Any] | None = None,
        max_turns: int = 20,
        text_chunks: TextChunks = "delta",
        history_store: HistoryStore | LoadHistory | None = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        # Plain functions are accepted too: they are wrapped with `@tool`.
        self.tools = [tool(item) if _is_plain_function(item) else item for item in tools]
        self._engine = Engine(
            client=client or Client(),
            model=model,
            prompt=prompt,
            tools=self.tools,
            model_parameters=dict(model_parameters or {}),
            max_turns=max_turns,
            text_chunks=text_chunks,
            history_store=history_store,
        )

    def execute(
        self,
        prompt: str | None = None,
        *,
        history: History | None = None,
        conversation_id: str | None = None,
        text_chunks: TextChunks | None = None,
    ) -> Run:
        """Start a run and stream its events: `async for event in agent.execute(...)`.

        To continue a conversation, either pass `history` (e.g. `RunFinished.messages` of a
        previous run, or its `messages_json()`), or a `conversation_id` to load it from the
        agent's `history_store`. `text_chunks` overrides the agent's setting for this run.
        """
        return self._engine.execute(
            prompt, history=history, conversation_id=conversation_id, text_chunks=text_chunks
        )

    async def run(
        self, prompt: str | None = None, *, history: History | None = None, conversation_id: str | None = None
    ) -> RunFinished:
        """Run to completion and return the final event (with `.text` and `.messages`)."""
        # Nobody reads the text as it streams here, so group it per message: fewer events.
        async for event in self.execute(
            prompt, history=history, conversation_id=conversation_id, text_chunks="message"
        ):
            if isinstance(event, RunFinished):
                return event
        raise RuntimeError("The run ended without a RunFinished event")


def _is_plain_function(item: object) -> TypeGuard[Callable[..., Any]]:
    return callable(item) and not isinstance(item, (Tool, BuiltinTool)) and not hasattr(item, "invoke")
