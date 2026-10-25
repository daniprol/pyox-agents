"""`Agent`: the main entry point."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pyox_agents._core import BuiltinTool, Client, Engine, Run, RunFinished
from pyox_agents.tools import Tool, tool


class Agent:
    """A model, a system prompt and tools.

    Create it once and reuse it: each `execute()` call has its own history,
    so many runs can happen at the same time.

        agent = Agent(model="gpt-5-mini", prompt="Be brief.", tools=[get_weather])
        async for event in agent.execute("Weather in Vigo?"):
            ...
    """

    def __init__(
        self,
        *,
        model: str,
        prompt: str = "",
        tools: Sequence[Tool | BuiltinTool | Callable[..., Any]] = (),
        client: Client | None = None,
        model_parameters: Mapping[str, Any] | None = None,
        max_turns: int = 20,
    ) -> None:
        self.model = model
        self.prompt = prompt
        # Plain functions are accepted too: they are wrapped with `@tool`.
        self.tools = [
            item if isinstance(item, (Tool, BuiltinTool)) or not callable(item) else tool(item) for item in tools
        ]
        self._engine = Engine(
            client=client or Client(),
            model=model,
            prompt=prompt,
            tools=self.tools,
            model_parameters=dict(model_parameters or {}),
            max_turns=max_turns,
        )

    def execute(self, prompt: str | None = None, *, history: Sequence[Mapping[str, Any]] | None = None) -> Run:
        """Start a run and stream its events: `async for event in agent.execute(...)`.

        Pass `history` (e.g. `RunFinished.messages` of a previous run) to continue a conversation.
        """
        return self._engine.execute(prompt, history=list(history) if history is not None else None)

    async def run(
        self, prompt: str | None = None, *, history: Sequence[Mapping[str, Any]] | None = None
    ) -> RunFinished:
        """Run to completion and return the final event (with `.text` and `.messages`)."""
        async for event in self.execute(prompt, history=history):
            if isinstance(event, RunFinished):
                return event
        raise RuntimeError("The run ended without a RunFinished event")
