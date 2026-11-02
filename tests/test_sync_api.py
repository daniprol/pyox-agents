"""The sync API: `agent.run_sync(...)` and `for event in agent.execute_sync(...)`, with no event loop."""

import _thread
import asyncio
import contextvars
import os
import threading
import time
from dataclasses import replace
from typing import Any

import pytest

from conftest import MakeAgent
from factories import error_response, sse_data, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI, StreamResponse
from pyox_agents import (
    Answer,
    ApiError,
    Hooks,
    InputRequested,
    RunContext,
    RunFinished,
    TextDelta,
    tool,
)

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="none")


def where_am_i() -> str:
    """The thread a tool runs on, by name, and whether an event loop runs there."""
    try:
        asyncio.get_running_loop()
        loop = "loop"
    except RuntimeError:
        loop = "no loop"
    return f"{threading.current_thread().name}, {loop}"


@tool
def sync_tool() -> str:
    """Where do sync tools run?"""
    return where_am_i()


@tool
async def async_tool() -> str:
    """Where do async tools run?"""
    return where_am_i()


@tool(run_in_thread=True)
def threaded_tool() -> str:
    """Where do `run_in_thread` tools run?"""
    return where_am_i()


def tool_results(result: RunFinished) -> list[Any]:
    return [message["content"] for message in result.messages if message["role"] == "tool"]


# Running ------------------------------------------------------------------------------------------


def test_run_sync_returns_the_answer(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hello!"))

    result = make_agent().run_sync("Hi")

    assert (result.text, result.turns) == ("Hello!", 1)


def test_execute_sync_streams_the_events(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hello!"))

    events = list(make_agent().execute_sync("Hi"))

    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "Hello!"
    assert isinstance(events[-1], RunFinished)


def test_run_sync_raises_the_run_errors(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(error_response(400, "Bad request"))

    with pytest.raises(ApiError, match="Bad request"):
        make_agent().run_sync("Hi")


# Where Python code runs -----------------------------------------------------------------------------


def test_sync_tools_run_on_the_calling_thread_and_async_ones_on_pyox_loop(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("sync_tool"), tool_call("async_tool"), tool_call("threaded_tool")),
        text_response("Done"),
    )
    caller = threading.current_thread().name

    result = make_agent(tools=[sync_tool, async_tool, threaded_tool]).run_sync("Where?")

    sync_place, async_place, threaded_place = tool_results(result)
    assert sync_place == f"{caller}, no loop"
    assert async_place == "pyox-agents-loop, loop"
    assert threaded_place.endswith("no loop")
    assert not threaded_place.startswith(caller)


async def test_run_sync_works_inside_a_running_event_loop(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    """Like a Jupyter cell: the loop is running, and blocking it for the run is fine."""
    fake_openai.queue(tool_calls_response(tool_call("sync_tool"), tool_call("async_tool")), text_response("Done"))

    result = make_agent(tools=[sync_tool, async_tool]).run_sync("Where?")

    assert result.text == "Done"
    assert tool_results(result)[1] == "pyox-agents-loop, loop"


async def test_sync_tools_of_async_runs_run_on_worker_threads(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    both_running = threading.Barrier(2, timeout=5)

    @tool
    def slow() -> str:
        """Waits for the other call: only calls running at the same time get past it."""
        both_running.wait()
        return where_am_i()

    fake_openai.queue(tool_calls_response(tool_call("slow"), tool_call("slow")), text_response("Done"))

    result = await make_agent(tools=[slow]).run("Twice")

    places = tool_results(result)
    assert len(places) == 2
    assert all(place.endswith("no loop") for place in places)  # worker threads, never the event loop


def test_hooks_and_handlers_run_on_the_calling_thread(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    seen: list[str] = []

    @tool
    def confirm(ctx: RunContext) -> str:
        """Ask before going on."""
        return ctx.ask_sync("Go on?").value

    def on_request(request: InputRequested) -> Answer:
        seen.append(f"on_request: {where_am_i()}")
        return Answer("yes")

    def before_run(ctx: RunContext) -> None:
        seen.append(f"before_run: {where_am_i()}")

    fake_openai.queue(tool_calls_response(tool_call("confirm")), text_response("Done"))
    caller = threading.current_thread().name

    result = make_agent(tools=[confirm], hooks=[Hooks(before_run=before_run)]).run_sync("Go", on_request=on_request)

    assert tool_results(result) == ["yes"]
    assert seen == [f"before_run: {caller}, no loop", f"on_request: {caller}, no loop"]


def test_python_code_sees_the_callers_context_variables(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    def whose_request() -> str:
        """The request id."""
        return request_id.get()

    fake_openai.queue(tool_calls_response(tool_call("whose_request")), text_response("Done"))
    request_id.set("req-1")

    result = make_agent(tools=[whose_request]).run_sync("Whose?")

    assert tool_results(result) == ["req-1"]


def test_many_threads_run_at_once(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(*[text_response("Hi") for _ in range(8)])
    agent = make_agent()
    results: list[str] = []

    def run() -> None:
        results.append(agent.run_sync("Hi").text)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == ["Hi"] * 8


# Human in the loop --------------------------------------------------------------------------------


def test_ask_sync_without_a_handler_stops_the_run_until_answered(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    @tool
    def book(ctx: RunContext, city: str) -> str:
        """Book a trip."""
        price = ctx.step_sync("price", lambda: len(city) * 100)
        answer = ctx.ask_sync({"question": f"Pay {price}?"})
        return f"booked {city} for {price}: {answer.value}"

    fake_openai.queue(tool_calls_response(tool_call("book", city="Vigo")), text_response("Booked"))
    agent = make_agent(tools=[book])

    stopped = agent.run_sync("Book Vigo")
    [pending] = stopped.pending
    result = agent.run_sync(history=stopped.messages, answers={pending.call_id: Answer("ok")})

    assert pending.data == {"question": "Pay 400?"}
    assert tool_results(result)[-1] == "booked Vigo for 400: ok"


async def test_ask_sync_is_for_sync_tools(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    async def wrong(ctx: RunContext) -> str:
        """Asks the wrong way."""
        return ctx.ask_sync("?").value

    fake_openai.queue(tool_calls_response(tool_call("wrong")), text_response("Done"))

    result = await make_agent(tools=[wrong]).run("Go")

    assert "in async tools use `await ctx.ask(...)`" in tool_results(result)[0]


# Reading ------------------------------------------------------------------------------------------


def test_batches_and_web_streams_work_with_for(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hello!"), text_response("Hello!"))
    agent = make_agent()

    batches = list(agent.execute_sync("Hi").batches())
    stream = "".join(agent.execute_sync("Hi", text_chunks="message").ai_sdk())

    assert isinstance(batches[-1][-1], RunFinished)
    deltas = [part["delta"] for part in sse_data(stream) if isinstance(part, dict) and part["type"] == "text-delta"]
    assert deltas == ["Hello!"]


async def test_each_run_is_read_its_own_way(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    agent = make_agent()

    with pytest.raises(TypeError, match="use execute_sync"):
        next(iter(agent.execute("Hi").batches()))
    with pytest.raises(TypeError, match="not `async for`"):
        await anext(aiter(agent.execute_sync("Hi").ai_sdk()))


def test_stopping_to_read_ends_the_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    ran: list[str] = []

    @tool
    def remember() -> str:
        """Remember."""
        ran.append("ran")
        return "ok"

    # The tool call arrives well after the text, so the run cannot get there before the reader stops.
    answer = tool_calls_response(tool_call("remember"), text_pieces=("Let me see",))
    fake_openai.queue(replace(answer, chunk_delay_seconds=0.05), text_response("Done"))

    with make_agent(tools=[remember]).execute_sync("Hi") as run:
        for event in run:
            assert isinstance(event, TextDelta)
            break

    assert ran == []
    assert len(fake_openai.requests) == 1


def test_stop_keeps_what_the_run_did(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    pieces = [{"choices": [{"delta": {"content": piece}}]} for piece in ("one ", "two ", "three")]
    fake_openai.queue(StreamResponse(pieces, chunk_delay_seconds=0.05))
    run = make_agent().execute_sync("Count")

    for event in run:
        if isinstance(event, TextDelta):
            run.stop()
        if isinstance(event, RunFinished):
            assert event.stop_reason == "stopped"
            assert event.text.startswith("one")


# Ctrl+C -------------------------------------------------------------------------------------------


def test_ctrl_c_while_waiting_ends_the_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(StreamResponse(text_response("Late").chunks, delay_seconds=5))
    threading.Timer(0.2, _thread.interrupt_main).start()  # Ctrl+C, 0.2 s from now

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        make_agent().run_sync("Hi")

    assert time.monotonic() - started < 3  # long before the answer would come (5 s)


def test_ctrl_c_inside_a_sync_tool_is_not_a_tool_error(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    def interrupted() -> str:
        """Gets Ctrl+C."""
        raise KeyboardInterrupt

    fake_openai.queue(tool_calls_response(tool_call("interrupted")), text_response("Done"))

    with pytest.raises(KeyboardInterrupt):
        make_agent(tools=[interrupted]).run_sync("Go")

    assert len(fake_openai.requests) == 1  # the model never saw it


# Fork ---------------------------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork()")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork() in a process with threads
def test_a_process_forked_after_a_run_gets_a_clear_error(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("Hi"))
    agent = make_agent()
    agent.run_sync("Hi")

    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:  # the child
        try:
            agent.run_sync("Hi")
            message = b"no error"
        except RuntimeError as error:
            message = str(error).encode()
        os.write(write_end, message)
        os._exit(0)
    os.close(write_end)
    os.waitpid(pid, 0)

    assert b"before this process was forked" in os.read(read_end, 1000)
