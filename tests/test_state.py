"""The run's state: `execute(state=...)`, `ctx.state`, `StateChanged` and `RunFinished.state`."""

from typing import Any

from conftest import MakeAgent
from factories import collect_events, events_of_type, run_finished, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import RunContext, StateChanged, ToolFinished, tool


@tool
def add_todo(ctx: RunContext, text: str) -> str:
    """Add a todo."""
    ctx.state["todos"].append(text)
    return "added"


@tool
def rename(ctx: RunContext, title: str) -> str:
    """Rename the list."""
    ctx.state = {**ctx.state, "title": title}
    return "renamed"


@tool
def read_todos(ctx: RunContext) -> str:
    """Read the todos."""
    return ", ".join(ctx.state["todos"])


@tool
def add_then_fail(ctx: RunContext) -> str:
    """Add a todo, then fail."""
    ctx.state["todos"].append("half done")
    raise RuntimeError("Something broke")


def tool_results(events: list[Any]) -> list[tuple[str, bool]]:
    return [(event.result, event.is_error) for event in events_of_type(events, ToolFinished)]


async def test_tools_change_the_state_and_the_app_sees_each_change(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("add_todo", text="pack")),
        tool_calls_response(tool_call("rename", title="Trip")),
        text_response("Done"),
    )
    state = {"todos": [], "title": "List"}

    events = await collect_events(make_agent(tools=[add_todo, rename]).execute("Plan", state=state))

    changes = events_of_type(events, StateChanged)
    assert [change.delta for change in changes] == [
        [{"op": "add", "path": "/todos/0", "value": "pack"}],
        [{"op": "replace", "path": "/title", "value": "Trip"}],
    ]
    assert changes[-1].state == {"todos": ["pack"], "title": "Trip"}
    assert changes[0].subagent_id is None
    result = run_finished(events)
    assert result.state == {"todos": ["pack"], "title": "Trip"}
    assert state == {"todos": [], "title": "List"}  # the app's own dict is never changed


async def test_tools_running_together_keep_each_other_changes(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("add_todo", text="pack"), tool_call("rename", title="Trip")),
        text_response("Done"),
    )

    result = await make_agent(tools=[add_todo, rename]).run("Plan", state={"todos": [], "title": "List"})

    assert result.state == {"todos": ["pack"], "title": "Trip"}


async def test_failed_tools_change_nothing(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("add_then_fail")), text_response("Oops"))

    events = await collect_events(make_agent(tools=[add_then_fail]).execute("Go", state={"todos": []}))

    assert events_of_type(events, StateChanged) == []
    assert run_finished(events).state == {"todos": []}


async def test_without_state_it_is_none(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    def look(ctx: RunContext) -> str:
        """Look at the state."""
        return repr(ctx.state)

    fake_openai.queue(tool_calls_response(tool_call("look")), text_response("ok"))

    events = await collect_events(make_agent(tools=[look]).execute("Go"))

    assert tool_results(events) == [("None", False)]
    assert run_finished(events).state is None


# Subagents ----------------------------------------------------------------------------------------


async def test_subagents_see_the_state_only_when_allowed(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    helper = make_agent(tools=[read_todos, add_todo])
    agent = make_agent(
        tools=[
            helper.as_tool("no_access", "Help"),
            helper.as_tool("reader", "Help", state="read"),
            helper.as_tool("writer", "Help", state="write"),
        ]
    )
    calls = [("no_access", "read_todos", {}), ("reader", "read_todos", {}), ("reader", "add_todo", {"text": "x"})]
    calls.append(("writer", "add_todo", {"text": "pack"}))
    for subagent, subagent_tool, arguments in calls:
        fake_openai.queue(
            tool_calls_response(tool_call(subagent, task="Help")),
            tool_calls_response(tool_call(subagent_tool, **arguments)),
            text_response("ok"),
        )
    fake_openai.queue(text_response("Done"))

    events = await collect_events(agent.execute("Go", state={"todos": ["buy"]}))

    subagent_results = [
        (event.result, event.is_error) for event in events_of_type(events, ToolFinished) if event.subagent_id
    ]
    no_access_result, no_access_failed = subagent_results[0]
    assert no_access_failed
    assert "cannot see the state" in no_access_result
    assert subagent_results[1] == ("buy", False)
    assert subagent_results[2] == ("This subagent can only read the state, not change it", True)
    assert subagent_results[3] == ("added", False)
    [change] = events_of_type(events, StateChanged)
    assert change.subagent_id == "call_0"
    assert run_finished(events).state == {"todos": ["buy", "pack"]}
