"""`Agent`: the main entry point."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar

from pyox_agents import _background_loop
from pyox_agents._core import Client, Engine, Run, RunFinished, SyncRun
from pyox_agents.hooks import Hooks
from pyox_agents.human_input import OnRequest, Reply
from pyox_agents.messages import History, HistoryStore, LoadHistory, Prompt
from pyox_agents.schema import output_format
from pyox_agents.skills import SkillLike, SkillsHook
from pyox_agents.subagents import StateAccess, Subagent, TalkTo
from pyox_agents.telemetry import Instrumentation, recorder_for, require_opentelemetry
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


OutputT = TypeVar("OutputT")


class Agent(Generic[OutputT]):
    """A model, its instructions (the system prompt) and tools.

    Create it once and reuse it: each `stream()` call has its own history,
    so many runs can happen at the same time.

        agent = Agent(model="gpt-5-mini", instructions="Be brief.", tools=[get_weather])
        async for event in agent.stream("Weather in Vigo?"):
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

    An agent can be another agent's tool: `agent.as_tool(...)` (see `pyox_agents.subagents`).

    `skills` are instructions and tools the model loads when a task needs them
    (see `pyox_agents.skills`).

    `output_type` asks for a structured answer: a dataclass, `TypedDict`, pydantic model or
    JSON schema dict, sent as a strict JSON schema. `RunFinished.output` has the answer as
    that type (`None` when the run stops before answering: see `stop_reason`):

        agent = Agent(model="gpt-5-mini", output_type=Ticket)
        ticket = (await agent.run("Printer on fire")).output  # a `Ticket`

    `name` names the agent in traces. `instrument` says if its runs are traced with OpenTelemetry
    (see `pyox_agents.telemetry`): `None` follows `pyox_agents.instrument()`, `True` traces with the
    global providers, `False` never traces, or give an `Instrumentation`.
    """

    def __init__(
        self,
        *,
        model: str,
        name: str | None = None,
        instructions: str = "",
        tools: Sequence[ToolLike | Subagent | Callable[..., Any]] = (),
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
        output_type: type[OutputT] | Mapping[str, Any] | None = None,
        instrument: Instrumentation | bool | None = None,
        skills: Sequence[SkillLike] = (),
    ) -> None:
        if instrument:
            require_opentelemetry()
        self._model = model
        self._name = name
        self._instrument = instrument
        self._instructions = instructions
        self._api: Api = api
        # Plain functions are accepted too: they are wrapped with `@tool`.
        self._tools: tuple[ToolLike | Subagent, ...] = tuple(_as_tool(item) for item in tools)
        # Skills bring their own tools and a hook that hides them until the model loads the skill.
        skills_hook = SkillsHook(skills) if skills else None
        self._skills = tuple(skills)
        self._output_type = output_type
        output_schema, output_parser = output_format(output_type) if output_type is not None else (None, None)
        self._engine = Engine(
            client=client or Client(),
            model=model,
            name=name,
            instructions=instructions,
            tools=[*self._tools, *(skills_hook.tools if skills_hook else ())],
            model_parameters=dict(model_parameters or {}),
            max_turns=max_turns,
            text_chunks=text_chunks,
            history_store=history_store,
            hooks=[skills_hook, *hooks] if skills_hook else list(hooks),
            api=api,
            store=store,
            encrypted_reasoning=encrypted_reasoning,
            unanswered=unanswered,
            output_schema=output_schema,
            output_parser=output_parser,
        )

    # Read-only: the engine copied these when the agent was made, so changing
    # them later would silently do nothing. Make a new `Agent` instead.

    @property
    def model(self) -> str:
        return self._model

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def api(self) -> Api:
        return self._api

    @property
    def tools(self) -> tuple[ToolLike | Subagent, ...]:
        return self._tools

    @property
    def skills(self) -> tuple[SkillLike, ...]:
        return self._skills

    @property
    def output_type(self) -> type[OutputT] | Mapping[str, Any] | None:
        return self._output_type

    def stream(
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
        tools: Sequence[ToolLike | Subagent | Callable[..., Any]] | None = None,
        frontend_tools: Sequence[Mapping[str, Any]] | None = None,
        state: Any = None,
    ) -> Run[OutputT]:
        """Start a run and stream its events: `async for event in agent.stream(...)`.

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

        `tools` adds tools for this run only, on top of the agent's (the same kinds as
        `Agent(tools=...)`): tools made for one user, such as an MCP server opened with their
        token. Their names must differ from the agent's tools.

        `frontend_tools` adds tools the app runs, for this run only: dicts with `name`,
        `description` and `parameters` (a JSON schema), as AG-UI clients send them.

        `state` is the run's state, shared with the app (usually a dict, anything JSON works).
        Tools read and change it as `ctx.state`; each change comes as a `StateChanged` event,
        and `RunFinished.state` has the final state. Nothing is stored: give it again next run.

        Read the run event by event (`async for event in run`), in batches of every event
        already waiting (`run.batches()`), or as a web stream (`run.ai_sdk()`, `run.ag_ui(...)`).
        """
        return self._start(
            prompt,
            history=history,
            conversation_id=conversation_id,
            text_chunks=text_chunks,
            deps=deps,
            previous_response_id=previous_response_id,
            answers=answers,
            on_request=on_request,
            tools=tools,
            frontend_tools=frontend_tools,
            state=state,
        )

    def stream_sync(
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
        tools: Sequence[ToolLike | Subagent | Callable[..., Any]] | None = None,
        frontend_tools: Sequence[Mapping[str, Any]] | None = None,
        state: Any = None,
    ) -> SyncRun[OutputT]:
        """`stream`, for code without an event loop: `for event in agent.stream_sync(...)`.

        Read it event by event, in batches (`run.batches()`) or as a web stream: in Django,
        `StreamingHttpResponse(run.ai_sdk(), headers=ai_sdk.HEADERS)`. The arguments are the
        same as `stream`'s.

        The run starts at the first read. Rust does the HTTP and the streaming meanwhile, with
        the GIL released, so other threads keep running. The run's sync tools, hooks, history
        stores and `on_request` run on the thread reading it, like any function you call
        (Django's ORM sees the view's transaction); async ones on pyox's own event loop thread;
        tools with `run_in_thread=True` on worker threads, in parallel.

        Stopping early (`break`, `run.close()`, leaving `with`, a client that went away) ends
        the run at once. `run.stop()` ends it at the next safe point and keeps what it did.
        """
        if _background_loop.is_current():
            raise RuntimeError("Inside an async tool, use `await agent.run(...)` or `agent.stream(...)`")
        return self._start(
            prompt,
            history=history,
            conversation_id=conversation_id,
            text_chunks=text_chunks,
            deps=deps,
            previous_response_id=previous_response_id,
            answers=answers,
            on_request=on_request,
            tools=tools,
            frontend_tools=frontend_tools,
            state=state,
            sync=True,
        )

    def _start(
        self,
        prompt: Prompt | None,
        *,
        answers: Mapping[str, Reply] | None,
        tools: Sequence[ToolLike | Subagent | Callable[..., Any]] | None,
        frontend_tools: Sequence[Mapping[str, Any]] | None,
        **options: Any,
    ) -> Any:
        return self._engine.start(
            prompt,
            answers=dict(answers) if answers else None,
            tools=[_as_tool(item) for item in tools] if tools else None,
            frontend_tools=[dict(tool) for tool in frontend_tools] if frontend_tools else None,
            telemetry=recorder_for(self._instrument),
            **options,
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
        tools: Sequence[ToolLike | Subagent | Callable[..., Any]] | None = None,
        frontend_tools: Sequence[Mapping[str, Any]] | None = None,
        state: Any = None,
    ) -> RunFinished[OutputT]:
        """Run to completion (or until tool calls wait for a person, see `.pending`) and return
        the final event (with `.text` and `.messages`)."""
        # Nobody reads the text as it streams here, so group it per message: fewer events.
        async for event in self.stream(
            prompt,
            history=history,
            conversation_id=conversation_id,
            text_chunks="message",
            deps=deps,
            previous_response_id=previous_response_id,
            answers=answers,
            on_request=on_request,
            tools=tools,
            frontend_tools=frontend_tools,
            state=state,
        ):
            if isinstance(event, RunFinished):
                return event
        raise RuntimeError("The run ended without a RunFinished event")

    def run_sync(
        self,
        prompt: Prompt | None = None,
        *,
        history: History | None = None,
        conversation_id: str | None = None,
        deps: Any = None,
        previous_response_id: str | None = None,
        answers: Mapping[str, Reply] | None = None,
        on_request: OnRequest | None = None,
        tools: Sequence[ToolLike | Subagent | Callable[..., Any]] | None = None,
        frontend_tools: Sequence[Mapping[str, Any]] | None = None,
        state: Any = None,
    ) -> RunFinished[OutputT]:
        """`run`, for code without an event loop: scripts, Django and Flask views, Celery tasks,
        notebooks. Blocks this thread until the run ends (see `stream_sync` for where the
        run's Python code runs). Ctrl+C raises `KeyboardInterrupt` and ends the run."""
        with self.stream_sync(
            prompt,
            history=history,
            conversation_id=conversation_id,
            text_chunks="message",
            deps=deps,
            previous_response_id=previous_response_id,
            answers=answers,
            on_request=on_request,
            tools=tools,
            frontend_tools=frontend_tools,
            state=state,
        ) as run:
            for event in run:
                if isinstance(event, RunFinished):
                    return event
        raise RuntimeError("The run ended without a RunFinished event")

    def as_tool(
        self,
        name: str,
        description: str,
        *,
        inherit_tools: bool = False,
        talk_to: TalkTo | None = None,
        state: StateAccess | None = None,
    ) -> Subagent:
        """This agent as a tool for another agent: a subagent (see `pyox_agents.subagents`).

            writer = Agent(model="gpt-5", tools=[researcher.as_tool("research", "Research a topic.")])

        The model calls it with a `task`, and reads the subagent's final answer as the result.

        - `inherit_tools=True` also gives the subagent the calling agent's tools (its own win
          when names clash).
        - `talk_to`: who the subagent's model can ask questions: nobody (`None`), the person
          (`"user"`, with an `ask_user` tool) or the calling agent (`"parent"`, with an
          `ask_parent` tool; the calling agent can then answer and follow up).
        - `state`: what the subagent's tools may do with the run's state: nothing (`None`),
          `"read"` or `"write"`.
        """
        return Subagent(self, name, description, inherit_tools=inherit_tools, talk_to=talk_to, state=state)


def _as_tool(item: ToolLike | Subagent | Callable[..., Any]) -> ToolLike | Subagent:
    """Tools as given; plain functions wrapped with `@tool`."""
    if isinstance(item, (Tool, Subagent)) or hasattr(item, "invoke"):
        return item  # ty: ignore[invalid-return-type]  (`hasattr` does not narrow to `ToolLike`)
    return tool(item) if callable(item) else item  # not a tool at all: the engine raises
