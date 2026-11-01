"""MCP tools (`pyox_agents.mcp`), against a real MCP server (`tests/mcp_server.py`).

The server runs in this process for most tests, and as a real subprocess over stdio and HTTP
for the transport tests. Tests that depend on how the server asks questions run on both sides
of MCP's stateless rewrite: 2026-07-28 (questions come back as `input_required` results) and
2025-11-25 (the server sends its question while the call waits).
"""

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mcp_types
import pytest

import mcp_server
from conftest import MakeAgent, Tracing
from factories import collect_events, events_of_type, run_finished, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI
from pyox_agents import Answer, Approve, Deny, InputRequested, Message, ToolFinished
from pyox_agents.mcp import ApprovalPolicy, McpError, McpToolError, McpTools

SERVER_SCRIPT = Path(__file__).parent / "mcp_server.py"

# The protocol version, and the `mcp.Client(mode=...)` that gets it.
PROTOCOLS = {"2026-07-28": "auto", "2025-11-25": "legacy"}

FLOOR_QUESTION = {"type": "mcp_elicitation", "server": "rooms", "mode": "form", "message": "Which floor?"}


def is_the_floor_question(data: dict[str, Any]) -> bool:
    """What the person sees when the server asks for a floor (`InputRequested.data`)."""
    without_schema = {key: value for key, value in data.items() if key != "schema"}
    return without_schema == FLOOR_QUESTION and data["schema"]["properties"] == {
        "floor": {"title": "Floor", "type": "integer"}
    }


@pytest.fixture(params=list(PROTOCOLS))
def protocol(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(autouse=True)
def no_bookings() -> None:
    mcp_server.bookings.clear()


# The connection is opened in each test (`async with rooms:`), not in the fixture: pytest-asyncio
# tears fixtures down in another task, and the `mcp` package needs both ends in the same one.


@pytest.fixture
def rooms(protocol: str) -> McpTools:
    """The test server in this process, on the `protocol` version."""
    return McpTools(mcp_server.server, mode=PROTOCOLS[protocol])


@pytest.fixture
def current_rooms() -> McpTools:
    """The test server in this process, for tests where the protocol version makes no difference."""
    return McpTools(mcp_server.server)


def tool_messages(messages: list[Message]) -> list[Any]:
    return [message for message in messages if message["role"] == "tool"]


def floor_answer(floor: int) -> Answer:
    return Answer({"floor": floor})


# Listing tools ------------------------------------------------------------------------------------


async def test_tools_keep_the_servers_names_descriptions_and_schemas(current_rooms: McpTools) -> None:
    async with current_rooms:
        add = next(tool for tool in current_rooms.tools if tool.name == "add")

    assert add.description == "Add two numbers."
    assert add.parameters == {
        "type": "object",
        "properties": {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}},
        "required": ["a", "b"],
        "title": "addArguments",
    }
    assert add.info.annotations == mcp_types.ToolAnnotations(read_only_hint=True)
    assert add.requires_approval is False


async def test_tool_names_get_the_prefix_and_only_characters_openai_accepts() -> None:
    async with McpTools(mcp_server.server, prefix="rooms", include={"add", "floor.plan"}) as rooms:
        assert [tool.name for tool in rooms.tools] == ["rooms_add", "rooms_floor_plan"]
        assert [tool.info.name for tool in rooms.tools] == ["add", "floor.plan"]


async def test_include_and_exclude_choose_the_tools() -> None:
    async with McpTools(mcp_server.server, exclude={"whoami", "trace_parent", "broken"}) as rooms:
        assert [tool.name for tool in rooms.tools] == ["add", "cancel_booking", "book_room", "floor_plan"]


@pytest.mark.parametrize(
    ("policy", "needing_approval"),
    [
        (False, set()),
        (True, {"add", "cancel_booking", "book_room", "floor.plan"}),
        ("unless_read_only", {"cancel_booking", "book_room"}),
        ({"cancel_booking"}, {"cancel_booking"}),
        (lambda info: info.name.startswith("cancel"), {"cancel_booking"}),
    ],
    ids=["none", "all", "unless_read_only", "names", "function"],
)
async def test_approval_policies(policy: ApprovalPolicy, needing_approval: set[str]) -> None:
    include = {"add", "cancel_booking", "book_room", "floor.plan"}
    async with McpTools(mcp_server.server, include=include, requires_approval=policy) as rooms:
        assert {tool.info.name for tool in rooms.tools if tool.requires_approval} == needing_approval


async def test_tools_need_a_connection(current_rooms: McpTools) -> None:
    with pytest.raises(McpError, match="async with"):
        current_rooms.tools  # noqa: B018 (reading it raises)

    async with current_rooms:
        pass
    with pytest.raises(McpError, match="Not connected"):
        await current_rooms.call_tool("add", {"a": 1, "b": 2})


async def test_after_disconnecting_the_model_reads_an_error(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, current_rooms: McpTools
) -> None:
    async with current_rooms:
        agent = make_agent(tools=current_rooms.tools)
    fake_openai.queue(tool_calls_response(tool_call("add", a=1, b=2)), text_response("Sorry"))

    result = await agent.run("Add 1 and 2")

    [answered] = tool_messages(result.messages)
    assert answered["is_error"] is True
    assert "Not connected to the MCP server" in answered["content"]


# Calling tools from an agent ----------------------------------------------------------------------


async def test_the_model_calls_a_tool(fake_openai: FakeOpenAI, make_agent: MakeAgent, rooms: McpTools) -> None:
    fake_openai.queue(tool_calls_response(tool_call("add", a=2, b=3)), text_response("5"))

    async with rooms:
        events = await collect_events(make_agent(tools=rooms.tools).execute("Add 2 and 3"))

    [finished] = events_of_type(events, ToolFinished)
    assert (finished.result, finished.is_error) == ("5", False)
    # Structured results go to the app, not to the model.
    assert finished.metadata == {"structured_content": {"result": 5}}
    assert tool_messages(fake_openai.requests[1].body["messages"]) == [
        {"role": "tool", "tool_call_id": "call_0", "content": "5"}
    ]
    assert run_finished(events).text == "5"


async def test_images_reach_the_model(fake_openai: FakeOpenAI, make_agent: MakeAgent, current_rooms: McpTools) -> None:
    fake_openai.queue(tool_calls_response(tool_call("floor_plan")), text_response("A tiny room"))

    async with current_rooms:
        await make_agent(tools=current_rooms.tools).run("Show me the floor plan")

    # Chat Completions tool messages only take text, so the image follows in a user message.
    image_message = fake_openai.requests[1].body["messages"][3]
    assert image_message["role"] == "user"
    assert image_message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,iVBORw0KGgo")


async def test_tool_errors_reach_the_model(fake_openai: FakeOpenAI, make_agent: MakeAgent, rooms: McpTools) -> None:
    fake_openai.queue(tool_calls_response(tool_call("broken")), text_response("Sorry"))

    async with rooms:
        result = await make_agent(tools=rooms.tools).run("Try it")

    [answered] = tool_messages(result.messages)
    assert answered["is_error"] is True
    assert "The room service is down" in answered["content"]


async def test_tools_not_marked_read_only_wait_for_approval(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("add", a=1, b=1), tool_call("cancel_booking", booking_id="b1")),
        text_response("Cancelled"),
    )
    async with McpTools(mcp_server.server, requires_approval="unless_read_only") as rooms:
        agent = make_agent(tools=rooms.tools)
        stopped = await agent.run("Add 1 and 1, then cancel b1")
        assert [(request.tool_name, request.kind) for request in stopped.pending] == [("cancel_booking", "approval")]

        result = await agent.run(history=stopped.messages, answers={"call_1": Approve()})

    assert tool_messages(result.new_messages)[0]["content"] == "Cancelled b1"


def test_sync_runs_use_servers_opened_with_with(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, rooms: McpTools
) -> None:
    fake_openai.queue(tool_calls_response(tool_call("add", a=2, b=3)), text_response("5"))

    with rooms:
        result = make_agent(tools=rooms.tools).run_sync("Add 2 and 3")

    assert tool_messages(result.messages)[0]["content"] == "5"


async def test_a_server_works_only_on_the_loop_it_was_opened_on(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, current_rooms: McpTools
) -> None:
    fake_openai.queue(tool_calls_response(tool_call("add", a=2, b=3)), text_response("Sorry"))

    with current_rooms:  # on pyox's own loop, for sync runs
        result = await make_agent(tools=current_rooms.tools).run("Add 2 and 3")

    assert "opened on another event loop" in tool_messages(result.messages)[0]["content"]


# Questions from the server (elicitation) ----------------------------------------------------------


async def test_questions_from_the_server_go_to_on_request(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, rooms: McpTools
) -> None:
    fake_openai.queue(tool_calls_response(tool_call("book_room", day="friday")), text_response("Booked"))
    asked: list[InputRequested] = []

    def pick_floor_3(request: InputRequested) -> Answer:
        asked.append(request)
        return floor_answer(3)

    async with rooms:
        result = await make_agent(tools=rooms.tools).run("Book a room on Friday", on_request=pick_floor_3)

    [request] = asked
    assert (request.tool_name, request.kind) == ("book_room", "input")
    assert is_the_floor_question(request.data)
    assert mcp_server.bookings == ["friday on floor 3"]
    assert tool_messages(result.new_messages)[-1]["content"] == "Booked friday on floor 3"


async def test_questions_from_the_server_stop_the_run_until_answered(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, rooms: McpTools
) -> None:
    fake_openai.queue(tool_calls_response(tool_call("book_room", day="friday")), text_response("Booked"))

    async with rooms:
        agent = make_agent(tools=rooms.tools)
        stopped = await agent.run("Book a room on Friday")

        [request] = stopped.pending
        assert request.call_id == "call_0"
        assert is_the_floor_question(request.data)
        assert mcp_server.bookings == []

        # A later run (another HTTP request, in a web app) brings the answer: the call runs again.
        result = await agent.run(history=stopped.messages, answers={"call_0": floor_answer(2)})

    assert mcp_server.bookings == ["friday on floor 2"]
    assert tool_messages(result.new_messages)[0]["content"] == "Booked friday on floor 2"
    assert result.text == "Booked"


async def test_parallel_calls_get_their_own_answers(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, rooms: McpTools
) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("book_room", day="monday"), tool_call("book_room", day="tuesday")),
        text_response("Booked both"),
    )

    async with rooms:
        agent = make_agent(tools=rooms.tools)
        stopped = await agent.run("Book Monday and Tuesday")
        assert sorted(request.call_id for request in stopped.pending) == ["call_0", "call_1"]

        await agent.run(history=stopped.messages, answers={"call_0": floor_answer(1), "call_1": floor_answer(4)})

    assert sorted(mcp_server.bookings) == ["monday on floor 1", "tuesday on floor 4"]


async def test_denying_a_question_ends_the_call(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, rooms: McpTools
) -> None:
    fake_openai.queue(tool_calls_response(tool_call("book_room", day="friday")), text_response("OK"))

    async with rooms:
        agent = make_agent(tools=rooms.tools)
        stopped = await agent.run("Book a room on Friday")
        result = await agent.run(history=stopped.messages, answers={"call_0": Deny("No rooms this week")})

    [answered] = tool_messages(result.new_messages)
    assert (answered["content"], answered["is_error"]) == ("No rooms this week", True)
    assert mcp_server.bookings == []


async def test_questions_outside_a_run_are_declined(rooms: McpTools) -> None:
    async with rooms:
        with pytest.raises(McpToolError, match="decline"):
            await rooms.call_tool("book_room", {"day": "friday"})


# Transports ---------------------------------------------------------------------------------------


async def test_stdio_servers(fake_openai: FakeOpenAI, make_agent: MakeAgent, protocol: str) -> None:
    fake_openai.queue(tool_calls_response(tool_call("book_room", day="friday")), text_response("Booked"))
    server = McpTools.stdio(sys.executable, [str(SERVER_SCRIPT), "stdio"], mode=PROTOCOLS[protocol])

    async with server as rooms:
        assert rooms.client.protocol_version == protocol
        agent = make_agent(tools=rooms.tools)
        stopped = await agent.run("Book a room on Friday")
        [request] = stopped.pending
        assert is_the_floor_question(request.data)

        result = await agent.run(history=stopped.messages, answers={"call_0": floor_answer(5)})

    assert tool_messages(result.new_messages)[0]["content"] == "Booked friday on floor 5"


@pytest.fixture(scope="module")
def http_server_url() -> Iterator[str]:
    """The test server as a real Streamable HTTP server, in a subprocess."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, str(SERVER_SCRIPT), "http", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.monotonic() + 20
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            if time.monotonic() > deadline or process.poll() is not None:
                process.kill()
                raise RuntimeError("The MCP test server did not start") from None
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    process.terminate()
    process.wait()


async def test_http_servers(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, http_server_url: str, protocol: str
) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("whoami"), tool_call("book_room", day="friday")),
        text_response("Done"),
    )
    server = McpTools.http(http_server_url, headers={"Authorization": "Bearer secret"}, mode=PROTOCOLS[protocol])

    async with server as rooms:
        assert rooms.client.protocol_version == protocol
        result = await make_agent(tools=rooms.tools).run("Who am I? Book Friday.", on_request=lambda _: floor_answer(6))

    whoami, booking = tool_messages(result.new_messages)
    assert whoami["content"] == "Bearer secret"
    assert booking["content"] == "Booked friday on floor 6"


async def test_each_user_gets_their_own_server(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, http_server_url: str
) -> None:
    # One agent for the app; each request opens the server with its user's token.
    agent = make_agent(instructions="Help the user.")
    results = []
    for token in ["ana", "ben"]:
        fake_openai.queue(tool_calls_response(tool_call("whoami")), text_response("Done"))
        async with McpTools.http(http_server_url, headers={"Authorization": f"Bearer {token}"}) as server:
            results.append(await agent.run("Who am I?", tools=server.tools))

    assert [tool_messages(result.new_messages)[0]["content"] for result in results] == ["Bearer ana", "Bearer ben"]


# Prompts, resources and tracing -------------------------------------------------------------------


async def test_prompts_and_resources(current_rooms: McpTools) -> None:
    async with current_rooms:
        assert await current_rooms.prompt("review", day="friday") == [
            {"role": "user", "content": "Review the bookings for friday."}
        ]
        assert await current_rooms.resource("rooms://rules") == ["No food in the rooms."]


async def test_calls_carry_the_runs_trace_to_the_server(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, tracing: Tracing
) -> None:
    fake_openai.queue(tool_calls_response(tool_call("trace_parent")), text_response("Traced"))

    # Over a real transport: in-process servers get no `_meta.traceparent`.
    async with McpTools.stdio(sys.executable, [str(SERVER_SCRIPT), "stdio"]) as rooms:
        result = await make_agent(tools=rooms.tools, instrument=tracing.instrumentation).run("Trace")

    traceparent = tool_messages(result.messages)[0]["content"]
    execute_tool = tracing.span("execute_tool trace_parent").context
    assert execute_tool is not None
    assert traceparent.split("-")[1] == f"{execute_tool.trace_id:032x}"
