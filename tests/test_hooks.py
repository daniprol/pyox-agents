"""Hooks (`Agent(hooks=[...])`) and the `RunContext` given to tools and hooks."""

from dataclasses import dataclass, field
from typing import Any

import fast_tools
import pytest

from conftest import MakeAgent
from factories import collect_events, events_of_type, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import (
    AssistantMessage,
    HookError,
    Hooks,
    Message,
    RunContext,
    RunFinished,
    ToolCall,
    ToolFinished,
    ToolResult,
    tool,
)


@dataclass
class Shop:
    """Stands in for what an app gives as `deps`: a database, the current user..."""

    orders: dict[str, list[str]] = field(default_factory=lambda: {"user-1": ["book", "lamp"]})


@tool
async def list_orders(ctx: RunContext[Shop], status: str = "open") -> str:
    """List the user's orders."""
    assert ctx.conversation_id is None
    return f"{ctx.tool_call_id} turn {ctx.turn} {status}: " + ", ".join(ctx.deps.orders["user-1"])


def add(a: int, b: int) -> int:
    return a + b


async def test_tools_asking_for_ctx_get_it(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("list_orders")), text_response("Done"))

    events = await collect_events(make_agent(tools=[list_orders, add]).execute("Orders?", deps=Shop()))

    assert list_orders.parameters["properties"] == {"status": {"type": "string"}}  # ctx is hidden from the model
    finished = events_of_type(events, ToolFinished)
    assert [(event.result, event.is_error) for event in finished] == [("call_0 turn 1 open: book, lamp", False)]


async def test_before_tool_call_changes_or_blocks_calls(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    def double_first_number(ctx: RunContext, call: ToolCall) -> ToolCall | None:
        if call["arguments"].get("a") == 0:
            raise PermissionError("Zero is not allowed")
        if call["name"] == "add":
            return {**call, "arguments": {**call["arguments"], "a": call["arguments"]["a"] * 2}}
        return None

    fake_openai.queue(
        tool_calls_response(tool_call("add", a=2, b=3), tool_call("add", a=0, b=1)), text_response("Done")
    )
    agent = make_agent(tools=[add], hooks=[Hooks(before_tool_call=double_first_number)])

    events = await collect_events(agent.execute("Add"))

    finished = events_of_type(events, ToolFinished)
    assert [(event.result, event.is_error) for event in finished] == [
        ("7", False),
        ("PermissionError: Zero is not allowed", True),
    ]


async def test_after_tool_call_changes_results(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    async def explain_errors(ctx: RunContext, call: ToolCall, result: ToolResult) -> ToolResult | None:
        if result["is_error"]:
            return {"content": f"{call['name']} failed, try other arguments", "is_error": True}
        return None

    fake_openai.queue(tool_calls_response(tool_call("add", a=1, b=2), tool_call("add", a=1)), text_response("Done"))
    agent = make_agent(tools=[add], hooks=[Hooks(after_tool_call=explain_errors)])

    result = await agent.run("Add")

    tool_messages = [message for message in result.messages if message["role"] == "tool"]
    assert [(message["content"], message["is_error"]) for message in tool_messages] == [
        ("3", False),
        ("add failed, try other arguments", True),
    ]


async def test_before_model_request_changes_only_the_request(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    def add_reminder(ctx: RunContext, messages: list[Message]) -> list[Message]:
        return [*messages, {"role": "user", "content": f"(reminder for turn {ctx.turn}: be brief)"}]

    fake_openai.queue(text_response("Hi"))
    agent = make_agent(hooks=[Hooks(before_model_request=add_reminder)])

    result = await agent.run("Hello")

    sent = fake_openai.requests[0].body["messages"]
    assert [message["content"] for message in sent] == ["Hello", "(reminder for turn 1: be brief)"]
    assert result.messages == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi", "tool_calls": [], "model": "test-model"},
    ]


async def test_after_model_response_changes_the_answer(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    def refuse_tools(ctx: RunContext, message: AssistantMessage) -> AssistantMessage:
        return {"role": "assistant", "content": "I will not use tools.", "tool_calls": []}

    fake_openai.queue(tool_calls_response(tool_call("add", a=1, b=2)))
    agent = make_agent(tools=[add], hooks=[Hooks(after_model_response=refuse_tools)])

    result = await agent.run("Add")

    assert (result.text, result.turns) == ("I will not use tools.", 1)


async def test_hook_objects_run_in_order_and_see_the_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    calls: list[str] = []

    class Recorder:
        """Any object with hook methods works; sync and async can be mixed."""

        def before_run(self, ctx: RunContext[Shop]) -> None:
            calls.append(f"before_run {ctx.conversation_id} {type(ctx.deps).__name__}")

        async def before_tool_call(self, ctx: RunContext, call: ToolCall) -> ToolCall:
            calls.append(f"recorder sees a={call['arguments']['a']}")
            return {**call, "arguments": {"a": 10, "b": 1}}

        async def after_run(self, ctx: RunContext, result: RunFinished) -> None:
            calls.append(f"after_run {result.text} turns={ctx.turn} tokens={ctx.usage.input_tokens}")

    def second(ctx: RunContext, call: ToolCall) -> None:
        calls.append(f"second sees a={call['arguments']['a']}")

    fake_openai.queue(tool_calls_response(tool_call("add", a=1, b=2)), text_response("11"))
    agent = make_agent(tools=[add], hooks=[Recorder(), Hooks(before_tool_call=second)])

    await agent.run("Add", deps=Shop())

    assert calls == [
        "before_run None Shop",
        "recorder sees a=1",
        "second sees a=10",
        "after_run 11 turns=2 tokens=20",
    ]


async def test_failing_hooks_stop_the_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    def no_runs(ctx: RunContext) -> None:
        raise PermissionError("Runs are disabled")

    def broken(ctx: RunContext, messages: list[Message]) -> Any:
        return "not messages"

    with pytest.raises(PermissionError, match="Runs are disabled"):
        await make_agent(hooks=[Hooks(before_run=no_runs)]).run("Hi")
    with pytest.raises(HookError, match="before_model_request hook returned an invalid value"):
        await make_agent(hooks=[Hooks(before_model_request=broken)]).run("Hi")
    assert fake_openai.requests == []


def test_objects_without_hooks_are_rejected(make_agent: MakeAgent) -> None:
    hooks: list[Any] = [object()]
    with pytest.raises(TypeError, match="has no hooks"):
        make_agent(hooks=hooks)


async def test_rust_hooks_block_calls_and_trim_requests(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    long_history: list[Message] = []
    for number in range(15):
        long_history.append({"role": "user", "content": f"question {number}"})
        long_history.append({"role": "assistant", "content": f"answer {number}"})
    fake_openai.queue(
        tool_calls_response(tool_call("count_words", text="short text"), tool_call("count_words", text="a" * 10_001)),
        text_response("Done"),
    )
    agent = make_agent(tools=[fast_tools.count_words], hooks=[fast_tools.guard])

    events = await collect_events(agent.execute("Count", history=long_history))

    finished = events_of_type(events, ToolFinished)
    assert [(event.result, event.is_error) for event in finished] == [
        ("2", False),
        ("The text is too long (10001 bytes, at most 10000)", True),
    ]
    sent = fake_openai.requests[0].body["messages"]
    assert len(sent) == 19  # the last 20 messages, starting at a user message
    assert sent[0] == {"role": "user", "content": "question 6"}
    run_finished = events_of_type(events, RunFinished)[0]
    assert len(run_finished.messages) == 35  # the history keeps everything
