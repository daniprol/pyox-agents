"""Subagents: agents called like tools (`agent.as_tool(...)`), their tools, questions and approvals."""

from typing import Any

import pytest

from conftest import MakeAgent
from factories import (
    collect_events,
    error_response,
    events_of_type,
    run_finished,
    text_response,
    tool_call,
    tool_calls_response,
)
from fake_openai import FakeOpenAI
from pyox_agents import (
    Answer,
    Approve,
    InputRequested,
    Message,
    RunContext,
    SubagentFinished,
    SubagentStarted,
    TextDelta,
    ToolFinished,
    ToolMessage,
    ToolStarted,
    tool,
)


@tool
def search(query: str) -> str:
    """Search the web."""
    return f"results for {query}"


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool(requires_approval=True)
def book(flight: str) -> str:
    """Book a flight."""
    return f"booked {flight}"


def tool_names(fake_openai: FakeOpenAI, request_index: int) -> list[str]:
    return [tool["function"]["name"] for tool in fake_openai.requests[request_index].body.get("tools", [])]


def sent_messages(fake_openai: FakeOpenAI, request_index: int) -> list[dict[str, Any]]:
    return fake_openai.requests[request_index].body["messages"]


async def test_the_subagent_answer_is_the_tool_result(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    researcher = make_agent(model="small-model", prompt="You research.", tools=[search])
    writer = make_agent(prompt="You write.", tools=[researcher.as_tool("research", "Research a topic")])
    fake_openai.queue(
        tool_calls_response(tool_call("research", task="Find facts about Vigo")),
        text_response("Vigo is in Galicia"),  # the subagent
        text_response("Here is your article"),
    )

    events = await collect_events(writer.execute("Write about Vigo"))

    # The subagent's request: its own model, prompt and tools, and the task as its first message.
    request = fake_openai.requests[1].body
    assert request["model"] == "small-model"
    assert request["messages"] == [
        {"role": "system", "content": "You research."},
        {"role": "user", "content": "Find facts about Vigo"},
    ]
    assert tool_names(fake_openai, 1) == ["search"]
    # The writer reads its answer as the tool result.
    assert sent_messages(fake_openai, 2)[-1] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "Vigo is in Galicia",
    }

    started, *_ = events_of_type(events, SubagentStarted)
    assert (started.call_id, started.name, started.task, started.subagent_id) == (
        "call_0",
        "research",
        "Find facts about Vigo",
        None,
    )
    [finished] = events_of_type(events, SubagentFinished)
    assert (finished.result, finished.is_error, finished.waiting) == ("Vigo is in Galicia", False, False)
    assert (finished.usage.requests, finished.usage.input_tokens) == (1, 10)
    # Its text comes in the same stream, marked with its id.
    assert [(delta.text, delta.subagent_id) for delta in events_of_type(events, TextDelta)] == [
        ("Vigo is i", "call_0"),
        ("n Galicia", "call_0"),
        ("Here is yo", None),
        ("ur article", None),
    ]
    assert [type(event).__name__ for event in events if not isinstance(event, TextDelta)] == [
        "ToolStarted",
        "SubagentStarted",
        "SubagentFinished",
        "ToolFinished",
        "RunFinished",
    ]
    result = run_finished(events)
    assert (result.usage.requests, result.usage.input_tokens) == (3, 30)  # the subagent counts too
    # The tool message keeps the subagent's conversation; the model never sees it.
    tool_message = result.messages[2]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == "Vigo is in Galicia"
    assert [message["role"] for message in tool_message.get("subagent", [])] == ["user", "assistant"]


async def test_subagent_tools_run_and_are_marked(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    researcher = make_agent(tools=[search])
    writer = make_agent(tools=[researcher.as_tool("research", "Research")])
    fake_openai.queue(
        tool_calls_response(tool_call("research", task="Vigo")),
        tool_calls_response(tool_call("search", query="Vigo")),
        text_response("Found it"),
        text_response("Done"),
    )

    events = await collect_events(writer.execute("Go"))

    started = events_of_type(events, ToolStarted)
    assert [(event.call_id, event.name, event.subagent_id) for event in started] == [
        ("call_0", "research", None),
        ("call_0/call_0", "search", "call_0"),  # ids stay unique in the stream
    ]
    finished = events_of_type(events, ToolFinished)
    assert [(event.call_id, event.result) for event in finished] == [
        ("call_0/call_0", "results for Vigo"),
        ("call_0", "Found it"),
    ]


async def test_subagents_can_inherit_the_calling_agent_tools(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool(name="add")  # the same name as the writer's `add`: its own wins
    def own_add(a: int, b: int) -> int:
        return a + b

    researcher = make_agent(tools=[own_add])
    writer = make_agent(tools=[search, add, researcher.as_tool("research", "Research", inherit_tools=True)])
    fake_openai.queue(tool_calls_response(tool_call("research", task="Vigo")), text_response("ok"), text_response("ok"))

    await writer.run("Go")

    assert tool_names(fake_openai, 1) == ["add", "search"]  # never `research` itself


async def test_a_subagent_cannot_have_subagents(make_agent: MakeAgent) -> None:
    inner = make_agent()
    middle = make_agent(tools=[inner.as_tool("inner", "Inner")])

    with pytest.raises(ValueError, match="cannot have subagents"):
        make_agent(tools=[middle.as_tool("middle", "Middle")])


async def test_subagent_errors_are_results_the_calling_agent_can_handle(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    researcher = make_agent()
    writer = make_agent(tools=[researcher.as_tool("research", "Research")])
    fake_openai.queue(
        tool_calls_response(tool_call("research", task="Vigo")),
        error_response(400, "Bad model"),
        text_response("Sorry"),
    )

    events = await collect_events(writer.execute("Go"))

    [finished] = events_of_type(events, SubagentFinished)
    assert finished.is_error
    assert "Bad model" in finished.result
    [tool_finished] = events_of_type(events, ToolFinished)
    assert tool_finished.is_error
    assert tool_finished.result.startswith("The subagent failed")
    assert run_finished(events).text == "Sorry"


# Approvals and questions go to the person ---------------------------------------------------------


async def test_subagent_approvals_stop_the_run_and_resume(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    travel_agent = make_agent(tools=[book])
    assistant = make_agent(tools=[travel_agent.as_tool("travel", "Book trips")])
    fake_openai.queue(
        tool_calls_response(tool_call("travel", task="Book F1")),
        tool_calls_response(tool_call("book", flight="F1")),
    )

    stopped = await assistant.run("Book F1")

    [request] = stopped.pending
    assert (request.call_id, request.subagent_id, request.kind, request.tool_name) == (
        "call_0/call_0",
        "call_0",
        "approval",
        "book",
    )
    waiting = stopped.messages[-1]
    assert waiting["role"] == "tool"
    assert waiting["content"] == ""  # no result yet
    assert waiting.get("subagent")  # its conversation, to continue it

    fake_openai.queue(text_response("Booked F1"), text_response("All done"))
    events = await collect_events(assistant.execute(history=stopped.messages, answers={"call_0/call_0": Approve()}))

    result = run_finished(events)
    assert (result.text, result.pending) == ("All done", [])
    assert sent_messages(fake_openai, 2)[-1] == {"role": "tool", "tool_call_id": "call_0", "content": "booked F1"}
    assert sent_messages(fake_openai, 3)[-1] == {"role": "tool", "tool_call_id": "call_0", "content": "Booked F1"}


async def test_answers_for_a_waiting_subagent_must_use_its_ids(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    travel_agent = make_agent(tools=[book])
    assistant = make_agent(tools=[travel_agent.as_tool("travel", "Book trips")])
    fake_openai.queue(
        tool_calls_response(tool_call("travel", task="Book F1")), tool_calls_response(tool_call("book", flight="F1"))
    )
    stopped = await assistant.run("Book F1")

    with pytest.raises(ValueError, match="waits for its subagent"):
        await assistant.run(history=stopped.messages, answers={"call_0": Approve()})
    with pytest.raises(ValueError, match="not waiting"):
        await assistant.run(history=stopped.messages, answers={"call_0/call_9": Approve()})


async def test_answers_need_the_subagent_conversation(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    travel_agent = make_agent(tools=[book])
    assistant = make_agent(tools=[travel_agent.as_tool("travel", "Book trips")])
    fake_openai.queue(
        tool_calls_response(tool_call("travel", task="Book F1")), tool_calls_response(tool_call("book", flight="F1"))
    )
    stopped = await assistant.run("Book F1")
    # A browser's copy of the history has the waiting call, but not the subagent's conversation.
    waiting: ToolMessage = {
        "role": "tool",
        "tool_call_id": "call_0",
        "name": "travel",
        "content": "",
        "requests": [{"kind": "approval", "data": None, "metadata": None, "answer": None}],
    }
    history: list[Message] = [*stopped.messages[:-1], waiting]

    with pytest.raises(ValueError, match="no conversation for this subagent"):
        await assistant.run(history=history, answers={"call_0/call_0": Approve()})


async def test_live_handlers_answer_subagent_requests(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    async def pick_seat(ctx: RunContext) -> str:
        """Pick a seat."""
        answer = await ctx.ask("Window or aisle?")
        return f"seat {answer.value}"

    asked: list[InputRequested] = []

    def on_request(request: InputRequested) -> Answer:
        asked.append(request)
        return Answer("window")

    travel_agent = make_agent(tools=[pick_seat])
    assistant = make_agent(tools=[travel_agent.as_tool("travel", "Book trips")])
    fake_openai.queue(
        tool_calls_response(tool_call("travel", task="Pick a seat")),
        tool_calls_response(tool_call("pick_seat")),
        text_response("Window seat"),
        text_response("Done"),
    )

    result = await assistant.run("Seat?", on_request=on_request)

    assert [(request.call_id, request.subagent_id, request.data) for request in asked] == [
        ("call_0/call_0", "call_0", "Window or aisle?")
    ]
    assert result.text == "Done"
    assert sent_messages(fake_openai, 2)[-1]["content"] == "seat window"


async def test_talk_to_user_lets_the_subagent_ask_the_person(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    planner = make_agent()
    assistant = make_agent(tools=[planner.as_tool("plan", "Plan a trip", talk_to="user")])
    fake_openai.queue(
        tool_calls_response(tool_call("plan", task="Plan a trip")),
        tool_calls_response(tool_call("ask_user", question="Which city?")),
    )

    stopped = await assistant.run("Plan my trip")

    assert tool_names(fake_openai, 1) == ["ask_user"]
    [request] = stopped.pending
    assert (request.call_id, request.kind, request.data) == ("call_0/call_0", "input", "Which city?")

    fake_openai.queue(text_response("3 days in Vigo"), text_response("Here is the plan"))
    result = await assistant.run(history=stopped.messages, answers={"call_0/call_0": Answer("Vigo")})

    assert result.text == "Here is the plan"
    assert sent_messages(fake_openai, 2)[-1] == {"role": "tool", "tool_call_id": "call_0", "content": "Vigo"}


async def test_talk_to_parent_goes_back_and_forth(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    planner = make_agent(prompt="You plan.")
    assistant = make_agent(tools=[planner.as_tool("plan", "Plan a trip", talk_to="parent")])
    fake_openai.queue(
        tool_calls_response(tool_call("plan", task="Plan a trip to Vigo")),
        tool_calls_response(tool_call("ask_parent", question="How many days?")),  # the subagent stops here
        tool_calls_response(tool_call("plan", task="3 days", subagent_id="call_0")),
        text_response("Day 1: Cies islands..."),  # the subagent goes on
        text_response("Here is the plan"),
    )

    result = await assistant.run("Plan my trip to Vigo")

    assert tool_names(fake_openai, 1) == ["ask_parent"]
    assert "subagent_id" in fake_openai.requests[0].body["tools"][0]["function"]["parameters"]["properties"]
    question = sent_messages(fake_openai, 2)[-1]["content"]
    assert question.startswith("The subagent asks: How many days?")
    assert 'subagent_id="call_0"' in question
    # The subagent continues its own conversation, with the answer as a new message.
    continued = sent_messages(fake_openai, 3)
    assert [message["role"] for message in continued] == ["system", "user", "assistant", "tool", "user"]
    assert continued[-1] == {"role": "user", "content": "3 days"}
    answer = sent_messages(fake_openai, 4)[-1]["content"]
    assert answer.startswith("Day 1: Cies islands...")
    assert "To follow up" in answer
    assert result.text == "Here is the plan"


async def test_continuing_needs_a_known_subagent_id(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    planner = make_agent()
    assistant = make_agent(tools=[planner.as_tool("plan", "Plan a trip", talk_to="parent")])
    fake_openai.queue(tool_calls_response(tool_call("plan", task="More", subagent_id="call_7")), text_response("Sorry"))

    await assistant.run("Plan")

    assert sent_messages(fake_openai, 1)[-1]["content"] == 'There is no conversation with subagent_id "call_7"'


async def test_subagents_running_together_keep_their_own_ids(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    researcher = make_agent(tools=[search])
    writer = make_agent(tools=[researcher.as_tool("research", "Research")])
    fake_openai.queue(
        tool_calls_response(tool_call("research", task="Vigo"), tool_call("research", task="Lugo")),
        text_response("found"),  # both subagents get the same answer: they run at the same time, in any order
        text_response("found"),
        text_response("Done"),
    )

    events = await collect_events(writer.execute("Go"))

    assert sorted((event.call_id, event.task) for event in events_of_type(events, SubagentStarted)) == [
        ("call_0", "Vigo"),
        ("call_1", "Lugo"),
    ]
    finished = events_of_type(events, SubagentFinished)
    assert sorted((event.call_id, event.result) for event in finished) == [("call_0", "found"), ("call_1", "found")]
    assert {delta.subagent_id for delta in events_of_type(events, TextDelta)} == {"call_0", "call_1", None}
    result = run_finished(events)
    assert result.usage.requests == 4
    assert sent_messages(fake_openai, 3)[-2:] == [
        {"role": "tool", "tool_call_id": "call_0", "content": "found"},
        {"role": "tool", "tool_call_id": "call_1", "content": "found"},
    ]


async def test_inherited_tools_keep_their_approval_rules(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    travel_agent = make_agent()
    assistant = make_agent(tools=[book, travel_agent.as_tool("travel", "Book trips", inherit_tools=True)])
    fake_openai.queue(
        tool_calls_response(tool_call("travel", task="Book F1")),
        tool_calls_response(tool_call("book", flight="F1")),
    )

    stopped = await assistant.run("Book F1")

    assert [(request.call_id, request.kind, request.tool_name) for request in stopped.pending] == [
        ("call_0/call_0", "approval", "book")
    ]


async def test_subagents_can_use_inherited_frontend_tools(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    location = {"name": "get_location", "description": "The user's location.", "parameters": {"type": "object"}}
    guide = make_agent()
    assistant = make_agent(tools=[guide.as_tool("guide", "Guide the user", inherit_tools=True)])
    fake_openai.queue(
        tool_calls_response(tool_call("guide", task="Where is the user?")),
        tool_calls_response(tool_call("get_location")),
    )

    stopped = await assistant.run("Guide me", frontend_tools=[location])

    assert tool_names(fake_openai, 1) == ["get_location"]
    [request] = stopped.pending
    assert (request.call_id, request.kind, request.tool_name) == ("call_0/call_0", "frontend", "get_location")

    fake_openai.queue(text_response("You are in Vigo"), text_response("Welcome to Vigo"))
    result = await assistant.run(
        history=stopped.messages, answers={"call_0/call_0": Answer("Vigo")}, frontend_tools=[location]
    )

    assert result.text == "Welcome to Vigo"
    assert sent_messages(fake_openai, 2)[-1]["content"] == "Vigo"  # the subagent reads the app's answer


async def test_subagent_tools_get_the_same_deps(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    def whoami(ctx: RunContext) -> str:
        """Say who the user is."""
        return ctx.deps["user"]

    helper = make_agent(tools=[whoami])
    agent = make_agent(tools=[helper.as_tool("helper", "Help")])
    fake_openai.queue(
        tool_calls_response(tool_call("helper", task="Who?")),
        tool_calls_response(tool_call("whoami")),
        text_response("Dani"),
        text_response("You are Dani"),
    )

    events = await collect_events(agent.execute("Who am I?", deps={"user": "Dani"}))

    subagent_tool, _ = events_of_type(events, ToolFinished)
    assert (subagent_tool.call_id, subagent_tool.result) == ("call_0/call_0", "Dani")


async def test_a_subagent_that_hits_max_turns_fails_its_tool(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    looper = make_agent(tools=[search], max_turns=1)
    agent = make_agent(tools=[looper.as_tool("loop", "Loop")])
    fake_openai.queue(
        tool_calls_response(tool_call("loop", task="Go")),
        tool_calls_response(tool_call("search", query="again")),
        text_response("It gave up"),
    )

    events = await collect_events(agent.execute("Go"))

    [finished] = events_of_type(events, SubagentFinished)
    assert finished.is_error
    assert "Stopped after 1 turns" in finished.result
    assert run_finished(events).text == "It gave up"
