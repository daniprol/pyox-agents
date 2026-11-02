"""What the model reads when a tool fails, and how apps and tools choose it."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from conftest import MakeAgent
from factories import collect_events, events_of_type, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import (
    HookError,
    Hooks,
    InvalidArguments,
    RunContext,
    StopRun,
    ToolCall,
    ToolFailed,
    ToolFinished,
    tool,
)


async def tool_results(fake_openai: FakeOpenAI, make_agent: MakeAgent, *calls: Any, **agent: Any) -> list[str]:
    """Runs one turn of tool calls, and returns what the model read from each."""
    fake_openai.queue(tool_calls_response(*calls), text_response("Done"))
    events = await collect_events(make_agent(**agent).execute("Go"))
    finished = events_of_type(events, ToolFinished)
    assert all(event.is_error for event in finished)
    return [event.result for event in finished]


@tool
def search_flights(origin: str, date: str) -> str:
    """Search flights."""
    raise ToolFailed(f"No flights from {origin} on {date}. Try another date.")


@tool
def read_config(path: str) -> str:
    """Read a config file."""
    raise FileNotFoundError(f"No such file: {path!r}")


async def test_tool_failed_gives_the_model_its_message_as_it_is(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    results = await tool_results(
        fake_openai,
        make_agent,
        tool_call("search_flights", origin="VGO", date="2026-12-24"),
        tool_call("read_config", path="app.yml"),
        tools=[search_flights, read_config],
    )

    assert results == [
        "No flights from VGO on 2026-12-24. Try another date.",
        "FileNotFoundError: No such file: 'app.yml'",
    ]


async def test_unexpected_tool_errors_are_logged(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, caplog: pytest.LogCaptureFixture
) -> None:
    await tool_results(
        fake_openai,
        make_agent,
        tool_call("search_flights", origin="VGO", date="today"),
        tool_call("read_config", path="app.yml"),
        tools=[search_flights, read_config],
    )

    # Only the unexpected one: a `ToolFailed` is the tool's own answer.
    [record] = [record for record in caplog.records if record.name == "pyox_agents"]
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "Tool 'read_config' failed"
    assert isinstance(record.exc_info, tuple)
    assert isinstance(record.exc_info[1], FileNotFoundError)


async def test_on_tool_error_chooses_what_the_model_reads(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    seen: list[tuple[str, type[Exception]]] = []

    async def on_tool_error(ctx: RunContext, call: ToolCall, error: Exception) -> str | None:
        seen.append((call["name"], type(error)))
        if isinstance(error, FileNotFoundError):
            return "The config is missing."
        return None  # the default text

    results = await tool_results(
        fake_openai,
        make_agent,
        tool_call("read_config", path="app.yml"),
        tool_call("search_flights", origin="VGO", date="today"),
        tool_call("nope"),
        tools=[read_config, search_flights],
        hooks=[Hooks(on_tool_error=on_tool_error)],
    )

    assert results == [
        "The config is missing.",
        "No flights from VGO on today. Try another date.",
        'Unknown tool "nope"',
    ]
    # The calls run at the same time, so the hook sees them in any order.
    assert sorted(seen, key=str) == sorted(
        [("read_config", FileNotFoundError), ("search_flights", ToolFailed), ("nope", ToolFailed)], key=str
    )


async def test_a_tools_own_on_error_comes_before_the_agents(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    def hide_paths(ctx: RunContext, call: ToolCall, error: Exception) -> str | None:
        return "The file could not be read." if isinstance(error, OSError) else None

    @tool(on_error=hide_paths)
    def read_secret(name: str) -> str:
        """Read a secret."""
        raise PermissionError(f"/etc/secrets/{name}")

    def agent_level(ctx: RunContext, call: ToolCall, error: Exception) -> str:
        return f"agent saw {call['name']}"

    results = await tool_results(
        fake_openai,
        make_agent,
        tool_call("read_secret", name="db"),
        tool_call("read_config", path="app.yml"),
        tools=[read_secret, read_config],
        hooks=[Hooks(on_tool_error=agent_level)],
    )

    # `hide_paths` answered for its tool; `read_config` has none, so the agent's hook decides.
    assert results == ["The file could not be read.", "agent saw read_config"]


async def test_stop_run_ends_the_run_with_a_history_that_can_continue(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    @tool
    def charge_card(amount: float) -> str:
        """Charge the card."""
        raise StopRun("Payments are down")

    @tool
    async def check_stock(item: str) -> str:
        """Check the stock."""
        await asyncio.sleep(0.05)  # still running when the other call stops the run
        return "3 left"

    fake_openai.queue(tool_calls_response(tool_call("charge_card", amount=10), tool_call("check_stock", item="tea")))

    with pytest.raises(StopRun, match="Payments are down") as raised:
        await make_agent(tools=[charge_card, check_stock]).run("Buy tea")

    # Every call has a result, so `history=error.messages` is a valid conversation.
    assert raised.value.messages is not None
    tool_messages = [message for message in raised.value.messages if message["role"] == "tool"]
    assert [(message["name"], message["content"]) for message in tool_messages] == [
        ("charge_card", "The run was stopped: Payments are down"),
        ("check_stock", "3 left"),
    ]
    assert len(fake_openai.requests) == 1


async def test_hooks_can_stop_the_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    def credentials_are_fatal(ctx: RunContext, call: ToolCall, error: Exception) -> None:
        if isinstance(error, PermissionError):
            raise StopRun("Missing credentials") from error

    @tool
    def read_mail() -> str:
        """Read mail."""
        raise PermissionError("no token")

    fake_openai.queue(tool_calls_response(tool_call("read_mail")))

    with pytest.raises(StopRun, match="Missing credentials") as raised:
        await make_agent(tools=[read_mail], hooks=[Hooks(on_tool_error=credentials_are_fatal)]).run("Go")
    assert isinstance(raised.value.__cause__, PermissionError)


async def test_a_raising_on_tool_error_ends_the_run_with_a_hook_error(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    def buggy(ctx: RunContext, call: ToolCall, error: Exception) -> None:
        raise KeyError("oops")

    fake_openai.queue(tool_calls_response(tool_call("read_config", path="x")))

    with pytest.raises(HookError, match="on_tool_error hook raised KeyError") as raised:
        await make_agent(tools=[read_config], hooks=[Hooks(on_tool_error=buggy)]).run("Go")
    assert isinstance(raised.value.__cause__, KeyError)


async def test_interrupts_from_tools_are_not_tool_errors(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    async def give_up() -> str:
        """Give up."""
        raise asyncio.CancelledError

    fake_openai.queue(tool_calls_response(tool_call("give_up")))

    with pytest.raises(asyncio.CancelledError):
        await make_agent(tools=[give_up]).run("Go")


async def test_slow_tools_time_out(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    cancelled = asyncio.Event()

    @tool(timeout_seconds=0.05)
    async def fetch_page(url: str) -> str:
        """Fetch a page."""
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never"

    results = await tool_results(
        fake_openai, make_agent, tool_call("fetch_page", url="https://example.com"), tools=[fetch_page]
    )

    assert results == ['TimeoutError: Tool "fetch_page" timed out after 0.05 seconds']
    await asyncio.wait_for(cancelled.wait(), 1)  # the tool was cancelled, not left running


def test_timeouts_must_be_positive(make_agent: MakeAgent) -> None:
    @tool(timeout_seconds=0)
    def instant() -> str:
        """Instant."""
        return ""

    with pytest.raises(ValueError, match="`timeout_seconds` must be a positive number"):
        make_agent(tools=[instant])


# Arguments that do not fit the tool --------------------------------------------------------------


class Trip(BaseModel):
    city: str
    days: int = Field(ge=1, le=7)


@dataclass
class Point:
    x: float
    y: float


@tool
def forecast(trip: Trip, units: Literal["metric", "imperial"] = "metric") -> str:
    """Weather forecast."""
    return f"{trip.city}: sunny for {trip.days} days ({units})"


@tool
def distance(start: Point, end: Point | None = None, tags: list[str] | None = None) -> float:
    """Distance from a point."""
    end = end or Point(0, 0)
    return ((start.x - end.x) ** 2 + (start.y - end.y) ** 2) ** 0.5


async def test_arguments_are_checked_and_the_model_reads_what_is_wrong(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    results = await tool_results(
        fake_openai,
        make_agent,
        tool_call("forecast", trip={"city": "Vigo", "days": 10}, units=5),
        tool_call("distance", start={"x": "1"}, tags=["a", 2], speed=3),
        tools=[forecast, distance],
    )

    assert results == [
        "Invalid arguments for forecast:\n"
        "- trip.days: Input should be less than or equal to 7 (got 10)\n"
        '- units: expected one of "metric", "imperial", got 5',
        "Invalid arguments for distance:\n"
        "- start.y: missing\n"
        '- start.x: expected a number, got "1"\n'
        "- tags[1]: expected a string, got 2\n"
        "- speed: not a parameter of this tool",
    ]


async def test_hooks_see_invalid_arguments_with_each_problem(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    problems: list[str] = []

    def on_tool_error(ctx: RunContext, call: ToolCall, error: Exception) -> None:
        if isinstance(error, InvalidArguments):
            problems.extend(error.errors)

    await tool_results(
        fake_openai,
        make_agent,
        tool_call("distance"),
        tools=[distance],
        hooks=[Hooks(on_tool_error=on_tool_error)],
    )

    assert problems == ["start: missing"]


async def test_valid_arguments_still_arrive_built(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(
            tool_call("distance", start={"x": 3, "y": 4}), tool_call("forecast", trip={"city": "Vigo", "days": 2})
        ),
        text_response("Done"),
    )

    events = await collect_events(make_agent(tools=[distance, forecast]).execute("Go"))

    assert [event.result for event in events_of_type(events, ToolFinished)] == [
        "5.0",
        "Vigo: sunny for 2 days (metric)",
    ]
