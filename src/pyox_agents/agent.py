"""`Agent`: the main entry point."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeGuard

from pyox_agents._core import BuiltinTool, Client, Engine, Run, RunFinished
from pyox_agents.hooks import Hooks
from pyox_agents.human_input import OnRequest, Reply
from pyox_agents.messages import History, HistoryStore, LoadHistory, Prompt
from pyox_agents.tools import Tool, ToolLike, tool

TextChunks = Literal["delta", "paragraph", "message"] | int
"""How streamed text is grouped into `TextDelta` events:

- `"delta"`: every piece the API sends (default, lowest latency).
- an int N: every N pieces.
- `"paragraph"`: each markdown paragraph; code blocks are never split.
- `"message"`: the whole text of each assistant message.

Bigger chunks mean fewer events, so less work for Python.
"""

Api = Literal["chat", "responses"]
"""Which OpenAI API the agent uses:

- `"chat"`: Chat Completions (default). Also spoken by most compatible servers.
- `"responses"`: the Responses API, with encrypted reasoning and stored responses.
"""

Unanswered = Literal["close", "raise"]
"""What happens to tool calls still waiting for a person when a new prompt comes instead of answers:

- `"close"`: each one gets a result telling the model the user did not answer (default).
- `"raise"`: `PendingRequestsError`.
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

    `hooks` are called at fixed points of each run, to watch or change it (see `Hooks`).

    `api="responses"` uses OpenAI's Responses API. By default each request is
    stateless (`store=False`): the model's encrypted reasoning is kept in the
    history and sent back unchanged, which also keeps the prompt cache working.
    Turn `encrypted_reasoning` off for models that do not reason (e.g. gpt-4.1).
    With `store=True`, OpenAI keeps each response and requests send only the
    new messages (`previous_response_id`).

    Tool calls can wait for a person: approvals, `ctx.ask(...)` questions and
    frontend tools (see `pyox_agents.human_input`). `unanswered` says what
    happens to calls still waiting when a new prompt comes instead of answers.
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
        hooks: Sequence[Hooks | object] = (),
        api: Api = "chat",
        store: bool = False,
        encrypted_reasoning: bool = True,
        unanswered: Unanswered = "close",
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.api = api
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
            hooks=list(hooks),
            api=api,
            store=store,
            encrypted_reasoning=encrypted_reasoning,
            unanswered=unanswered,
        )

    def execute(
        self,
        prompt: Prompt | None = None,
        *,
        history: History | None = None,
        conversation_id: str | None = None,
        text_chunks: TextChunks | None = None,
        deps: Any = None,
        previous_response_id: str | None = None,
        answers: Mapping[str, Reply] | None = None,
        on_request: OnRequest | None = None,
        frontend_tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> Run:
        """Start a run and stream its events: `async for event in agent.execute(...)`.

        `prompt` is text, or a list of text, images and files: `["What is this?", image("cat.png")]`.

        To continue a conversation, either pass `history` (e.g. `RunFinished.messages` of a
        previous run, or its `messages_json()`), or a `conversation_id` to load it from the
        agent's `history_store`. `text_chunks` overrides the agent's setting for this run.
        `deps` is anything tools and hooks need (a database pool, the user...): they get it
        as `ctx.deps`.

        With `api="responses"` and `store=True`, a conversation OpenAI stored can also continue
        from its last response alone: `previous_response_id=result.response_id`.

        When tool calls wait for a person, the run stops and `RunFinished.pending` lists their
        requests. Continue with the same history and `answers={call_id: Answer(...) | Approve(...)
        | Deny(...)}` (no prompt). Or give `on_request`, a function (sync or async) that replies to
        each request while the run waits; returning `None` stops the run as without it.

        `frontend_tools` adds tools the app runs, for this run only: dicts with `name`,
        `description` and `parameters` (a JSON schema), as AG-UI clients send them.

        Read the run event by event (`async for event in run`), in batches of every event
        already waiting (`run.batches()`), or as a web stream (`run.ai_sdk()`, `run.ag_ui(...)`).
        """
        return self._engine.execute(
            prompt,
            history=history,
            conversation_id=conversation_id,
            text_chunks=text_chunks,
            deps=deps,
            previous_response_id=previous_response_id,
            answers=dict(answers) if answers else None,
            on_request=on_request,
            frontend_tools=[dict(tool) for tool in frontend_tools] if frontend_tools else None,
        )

    async def run(
        self,
        prompt: Prompt | None = None,
        *,
        history: History | None = None,
        conversation_id: str | None = None,
        deps: Any = None,
        previous_response_id: str | None = None,
        answers: Mapping[str, Reply] | None = None,
        on_request: OnRequest | None = None,
        frontend_tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> RunFinished:
        """Run to completion (or until tool calls wait for a person, see `.pending`) and return
        the final event (with `.text` and `.messages`)."""
        # Nobody reads the text as it streams here, so group it per message: fewer events.
        async for event in self.execute(
            prompt,
            history=history,
            conversation_id=conversation_id,
            text_chunks="message",
            deps=deps,
            previous_response_id=previous_response_id,
            answers=answers,
            on_request=on_request,
            frontend_tools=frontend_tools,
        ):
            if isinstance(event, RunFinished):
                return event
        raise RuntimeError("The run ended without a RunFinished event")


def _is_plain_function(item: object) -> TypeGuard[Callable[..., Any]]:
    return callable(item) and not isinstance(item, (Tool, BuiltinTool)) and not hasattr(item, "invoke")
