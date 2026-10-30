"""Human in the loop: tool approvals, `ctx.ask`, `ctx.step`, frontend tools, live and stop-and-resume runs."""

import json
from typing import Any

import pytest

from conftest import MakeAgent
from factories import (
    collect_events,
    events_of_type,
    responses_answer,
    text_response,
    tool_call,
    tool_calls_response,
)
from fake_openai import FakeOpenAI
from pyox_agents import (
    Answer,
    ApprovalRequired,
    Approve,
    Client,
    Deny,
    Hooks,
    InputRequested,
    Message,
    PendingRequestsError,
    RunContext,
    RunFinished,
    Tool,
    ToolCall,
    ToolFinished,
    ToolReturn,
    tool,
)


@tool(requires_approval=True)
def delete_file(path: str) -> str:
    """Delete a file."""
    return f"deleted {path}"


def tool_messages(messages: list[Message]) -> list[Any]:
    return [message for message in messages if message["role"] == "tool"]


def sent_tool_messages(fake_openai: FakeOpenAI, request_index: int) -> list[dict[str, Any]]:
    """The tool messages of one request as the model received them."""
    return [message for message in fake_openai.requests[request_index].body["messages"] if message["role"] == "tool"]


# Approvals ----------------------------------------------------------------------------------------


async def test_a_tool_needing_approval_stops_the_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")))

    events = await collect_events(make_agent(tools=[delete_file]).execute("Delete a.txt"))

    [request] = events_of_type(events, InputRequested)
    assert (request.call_id, request.tool_name, request.kind) == ("call_0", "delete_file", "approval")
    assert request.arguments == {"path": "a.txt"}
    assert events_of_type(events, ToolFinished) == []  # the waiting call has no result yet
    result = events[-1]
    assert isinstance(result, RunFinished)
    assert [pending.call_id for pending in result.pending] == ["call_0"]
    assert result.turns == 1
    [waiting] = tool_messages(result.messages)
    assert waiting["requests"] == [{"kind": "approval", "data": None, "metadata": None, "answer": None}]


async def test_approved_calls_run_and_the_run_continues(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")), text_response("Done"))
    agent = make_agent(tools=[delete_file])
    stopped = await agent.run("Delete a.txt")

    result = await agent.run(history=stopped.messages, answers={"call_0": Approve()})

    assert (result.text, result.pending) == ("Done", [])
    assert sent_tool_messages(fake_openai, 1) == [
        {"role": "tool", "tool_call_id": "call_0", "content": "deleted a.txt"}
    ]
    # The history only grows: the waiting message stays, the answered one follows it.
    waiting, answered = tool_messages(result.messages)
    assert waiting["requests"][0]["answer"] is None
    assert answered["content"] == "deleted a.txt"
    assert answered["requests"][0]["answer"] == {"type": "approve", "metadata": None}
    assert result.new_messages[0] == answered


async def test_denied_calls_give_the_model_an_error(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")), text_response("OK"))
    agent = make_agent(tools=[delete_file])
    stopped = await agent.run("Delete a.txt")

    result = await agent.run(history=stopped.messages, answers={"call_0": Deny("Keep that file")})

    [answered] = tool_messages(result.new_messages)
    assert (answered["content"], answered["is_error"]) == ("Keep that file", True)


async def test_approval_can_change_the_arguments(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")), text_response("Done"))
    agent = make_agent(tools=[delete_file])
    stopped = await agent.run("Delete a.txt")

    result = await agent.run(history=stopped.messages, answers={"call_0": Approve(arguments={"path": "b.txt"})})

    assert tool_messages(result.new_messages)[0]["content"] == "deleted b.txt"


async def test_a_function_decides_which_calls_need_approval(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    def outside_tmp(ctx: RunContext, arguments: dict[str, Any]) -> bool:
        return not arguments["path"].startswith("/tmp/")

    @tool(requires_approval=outside_tmp)
    def remove(path: str) -> str:
        return f"removed {path}"

    fake_openai.queue(tool_calls_response(tool_call("remove", path="/tmp/x"), tool_call("remove", path="/etc/y")))

    result = await make_agent(tools=[remove]).run("Clean up")

    assert [request.call_id for request in result.pending] == ["call_1"]
    done, waiting = tool_messages(result.messages)
    assert (done["content"], waiting["content"]) == ("removed /tmp/x", "")


async def test_hooks_can_ask_for_approval(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    calls = []

    def run(command: str) -> str:
        calls.append(command)
        return "ok"

    def guard(ctx: RunContext, call: ToolCall) -> None:
        if "rm " in call["arguments"]["command"]:
            raise ApprovalRequired({"command": call["arguments"]["command"]}, metadata={"risk": "high"})

    fake_openai.queue(tool_calls_response(tool_call("run", command="rm -rf build")), text_response("Done"))
    agent = make_agent(tools=[run], hooks=[Hooks(before_tool_call=guard)])
    stopped = await agent.run("Clean")

    [request] = stopped.pending
    assert (request.kind, request.data, request.metadata) == ("approval", {"command": "rm -rf build"}, {"risk": "high"})
    assert calls == []

    # On resume, the hook raises again but the stored approval is used.
    result = await agent.run(history=stopped.messages, answers={"call_0": Approve()})

    assert (result.text, calls) == ("Done", ["rm -rf build"])


async def test_tools_see_the_approval_metadata(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    signed_by = []

    @tool(requires_approval=True)
    def send_email(ctx: RunContext, to: str) -> str:
        approval = ctx.approval
        assert isinstance(approval, Approve)
        signed_by.append(approval.metadata["user"])
        return f"sent to {to}"

    fake_openai.queue(tool_calls_response(tool_call("send_email", to="bob")), text_response("Sent"))
    agent = make_agent(tools=[send_email])
    stopped = await agent.run("Email Bob")

    result = await agent.run(history=stopped.messages, answers={"call_0": Approve(metadata={"user": "dani"})})

    assert (tool_messages(result.new_messages)[0]["content"], signed_by) == ("sent to bob", ["dani"])
    assert "dani" not in json.dumps(fake_openai.requests[1].body)  # metadata never reaches the model


async def test_replies_must_fit_the_request(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")))
    agent = make_agent(tools=[delete_file])
    stopped = await agent.run("Delete a.txt")

    with pytest.raises(TypeError, match="Approve"):
        await agent.run(history=stopped.messages, answers={"call_0": Answer("yes")})


# Questions (ctx.ask) in stop-and-resume mode ------------------------------------------------------


class Bookings:
    """Counts the side effects a tool makes, to show what runs again on resume."""

    def __init__(self) -> None:
        self.searches = 0
        self.booked: list[str] = []


@tool
async def book_flight(ctx: RunContext[Bookings], city: str) -> str:
    """Book a flight, asking the user to pick one."""
    ctx.deps.searches += 1  # runs again on resume: put side effects after the question or in ctx.step
    options = [f"{city} 9:00", f"{city} 18:00"]
    answer = await ctx.ask({"options": options}, metadata={"widget": "flight-picker"})
    ctx.deps.booked.append(answer.value)
    return f"Booked {answer.value}"


async def test_ask_stops_the_run_and_the_answer_resumes_the_tool(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    fake_openai.queue(tool_calls_response(tool_call("book_flight", city="Paris")), text_response("Booked"))
    agent = make_agent(tools=[book_flight])
    bookings = Bookings()
    stopped = await agent.run("Fly me to Paris", deps=bookings)

    [request] = stopped.pending
    assert (request.kind, request.data, request.metadata) == (
        "input",
        {"options": ["Paris 9:00", "Paris 18:00"]},
        {"widget": "flight-picker"},
    )
    assert bookings.booked == []

    result = await agent.run(
        history=stopped.messages, answers={"call_0": Answer("Paris 18:00", metadata={"seat": "12A"})}, deps=bookings
    )

    assert result.text == "Booked"
    assert (bookings.searches, bookings.booked) == (2, ["Paris 18:00"])
    assert sent_tool_messages(fake_openai, 1) == [
        {"role": "tool", "tool_call_id": "call_0", "content": "Booked Paris 18:00"}
    ]
    assert "12A" not in json.dumps(fake_openai.requests[1].body)


async def test_a_denied_question_stops_the_tool(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("book_flight", city="Paris")), text_response("OK"))
    agent = make_agent(tools=[book_flight])
    bookings = Bookings()
    stopped = await agent.run("Fly me to Paris", deps=bookings)

    result = await agent.run(history=stopped.messages, answers={"call_0": Deny("Not today")}, deps=bookings)

    [answered] = tool_messages(result.new_messages)
    assert (answered["content"], answered["is_error"], bookings.booked) == ("Not today", True, [])


async def test_a_tool_can_ask_several_questions_over_several_runs(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    @tool
    async def sign_up(ctx: RunContext) -> str:
        name = await ctx.ask("Your name?", key="name")
        email = await ctx.ask("Your email?", key="email")
        return f"{name.value} <{email.value}>"

    fake_openai.queue(tool_calls_response(tool_call("sign_up")), text_response("Welcome"))
    agent = make_agent(tools=[sign_up])

    first = await agent.run("Sign me up")
    assert [(request.key, request.data) for request in first.pending] == [("name", "Your name?")]
    second = await agent.run(history=first.messages, answers={"call_0": Answer("Dani")})
    assert [(request.key, request.data) for request in second.pending] == [("email", "Your email?")]
    assert second.turns == 0  # no model request: the call still waits
    third = await agent.run(history=second.messages, answers={"call_0": Answer("dani@example.com")})

    assert third.text == "Welcome"
    assert tool_messages(third.new_messages)[0]["content"] == "Dani <dani@example.com>"
    assert len(fake_openai.requests) == 2


async def test_tool_results_can_carry_hidden_metadata(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    def find_user(name: str) -> ToolReturn:
        return ToolReturn(f"Found {name}", metadata={"user_id": 42})

    fake_openai.queue(tool_calls_response(tool_call("find_user", name="Ana")), text_response("Done"))

    events = await collect_events(make_agent(tools=[find_user]).execute("Find Ana"))

    [finished] = events_of_type(events, ToolFinished)
    assert (finished.result, finished.metadata) == ("Found Ana", {"user_id": 42})
    result = events[-1]
    assert isinstance(result, RunFinished)
    assert tool_messages(result.messages)[0]["metadata"] == {"user_id": 42}
    assert sent_tool_messages(fake_openai, 1) == [{"role": "tool", "tool_call_id": "call_0", "content": "Found Ana"}]


# Live mode (on_request) ---------------------------------------------------------------------------


async def test_on_request_replies_while_the_tool_waits(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    seen: list[InputRequested] = []

    async def pick_the_last(request: InputRequested) -> Answer:
        seen.append(request)
        return Answer(request.data["options"][-1])

    fake_openai.queue(tool_calls_response(tool_call("book_flight", city="Rome")), text_response("Booked"))
    bookings = Bookings()

    events = await collect_events(
        make_agent(tools=[book_flight]).execute("Fly me to Rome", deps=bookings, on_request=pick_the_last)
    )

    assert [request.call_id for request in seen] == ["call_0"]
    assert len(events_of_type(events, InputRequested)) == 1  # the request is also streamed
    assert (bookings.searches, bookings.booked) == (1, ["Rome 18:00"])  # the tool went on in place
    result = events[-1]
    assert isinstance(result, RunFinished)
    assert (result.text, result.pending) == ("Booked", [])


async def test_on_request_approves_calls(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("delete_file", path="a.txt"), tool_call("delete_file", path="b.txt")),
        text_response("Done"),
    )

    def only_a(request: InputRequested) -> Approve | Deny:
        return Approve() if request.arguments["path"] == "a.txt" else Deny()

    result = await make_agent(tools=[delete_file]).run("Delete", on_request=only_a)

    assert [(message["content"], message["is_error"]) for message in tool_messages(result.messages)] == [
        ("deleted a.txt", False),
        ("The user denied this tool call", True),
    ]


async def test_on_request_can_leave_requests_for_later(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")), text_response("Done"))
    agent = make_agent(tools=[delete_file])

    stopped = await agent.run("Delete a.txt", on_request=lambda request: None)
    result = await agent.run(history=stopped.messages, answers={"call_0": Approve()})

    assert [request.call_id for request in stopped.pending] == ["call_0"]
    assert result.text == "Done"


# Concurrent calls ---------------------------------------------------------------------------------


async def test_other_calls_finish_while_one_waits(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    runs = []

    @tool
    def add(a: int, b: int) -> int:
        runs.append("add")
        return a + b

    fake_openai.queue(
        tool_calls_response(tool_call("add", a=1, b=2), tool_call("delete_file", path="a.txt")),
        text_response("Done"),
    )
    agent = make_agent(tools=[add, delete_file])
    stopped = await agent.run("Go")

    result = await agent.run(history=stopped.messages, answers={"call_1": Approve()})

    assert result.text == "Done"
    assert runs == ["add"]  # finished calls are not run again
    assert sent_tool_messages(fake_openai, 1) == [
        {"role": "tool", "tool_call_id": "call_0", "content": "3"},
        {"role": "tool", "tool_call_id": "call_1", "content": "deleted a.txt"},
    ]


async def test_answers_must_name_waiting_calls(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")))
    agent = make_agent(tools=[delete_file])
    stopped = await agent.run("Delete a.txt")

    with pytest.raises(ValueError, match="call_9"):
        await agent.run(history=stopped.messages, answers={"call_9": Approve()})


async def test_answers_and_a_prompt_cannot_be_mixed(make_agent: MakeAgent) -> None:
    with pytest.raises(ValueError, match="not both"):
        await make_agent().run("Hi", answers={"call_0": Approve()})


# Steps (ctx.step) ---------------------------------------------------------------------------------


async def test_steps_run_once_across_resumes(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    charges: list[int] = []

    async def charge(amount: int) -> dict[str, Any]:
        charges.append(amount)
        return {"receipt": f"r-{amount}"}

    @tool
    async def buy(ctx: RunContext, amount: int) -> str:
        receipt = await ctx.step("charge", charge, amount)
        answer = await ctx.ask(f"Send receipt {receipt['receipt']} by email?")
        return f"{receipt['receipt']} sent: {answer.value}"

    fake_openai.queue(tool_calls_response(tool_call("buy", amount=30)), text_response("Done"))
    agent = make_agent(tools=[buy])
    stopped = await agent.run("Buy")

    assert tool_messages(stopped.messages)[0]["steps"] == {"charge": {"receipt": "r-30"}}
    result = await agent.run(history=stopped.messages, answers={"call_0": Answer("yes")})

    assert charges == [30]
    assert tool_messages(result.new_messages)[0]["content"] == "r-30 sent: yes"


async def test_step_names_must_be_unique(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    async def twice(ctx: RunContext) -> str:
        await ctx.step("same", lambda: 1)
        await ctx.step("same", lambda: 2)
        return "never"

    fake_openai.queue(tool_calls_response(tool_call("twice")), text_response("Sorry"))

    result = await make_agent(tools=[twice]).run("Go")

    [message] = tool_messages(result.messages)
    assert message["is_error"]
    assert "same" in message["content"]


# Frontend tools -----------------------------------------------------------------------------------


@tool(frontend=True)
def pick_color(options: list[str]) -> str:
    """Let the user pick a color in the app."""
    raise NotImplementedError


get_location = Tool.frontend_tool(
    "get_location", "Get the user's location.", {"type": "object", "properties": {}, "required": []}
)


async def test_frontend_tools_wait_for_the_app(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("pick_color", options=["red", "blue"]), tool_call("get_location")),
        text_response("Nice"),
    )
    agent = make_agent(tools=[pick_color, get_location])
    stopped = await agent.run("Pick a color")

    assert [(request.kind, request.data) for request in stopped.pending] == [
        ("frontend", {"options": ["red", "blue"]}),
        ("frontend", {}),
    ]
    assert [schema["function"]["name"] for schema in fake_openai.requests[0].body["tools"]] == [
        "pick_color",
        "get_location",
    ]

    result = await agent.run(
        history=stopped.messages,
        answers={"call_0": Answer("blue"), "call_1": Answer({"city": "Vigo"})},
    )

    assert sent_tool_messages(fake_openai, 1) == [
        {"role": "tool", "tool_call_id": "call_0", "content": "blue"},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"city":"Vigo"}'},
    ]
    assert result.text == "Nice"


async def test_frontend_tools_can_be_answered_live(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("get_location")), text_response("OK"))

    result = await make_agent(tools=[get_location]).run("Where am I?", on_request=lambda request: Deny("No access"))

    [message] = tool_messages(result.messages)
    assert (message["content"], message["is_error"]) == ("No access", True)


# Unanswered calls ---------------------------------------------------------------------------------


async def test_a_new_prompt_closes_waiting_calls(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")), text_response("Never mind"))
    agent = make_agent(tools=[delete_file])
    stopped = await agent.run("Delete a.txt")

    result = await agent.run("Actually, forget it", history=stopped.messages)

    assert result.text == "Never mind"
    sent = fake_openai.requests[1].body["messages"]
    assert [message["role"] for message in sent] == ["user", "assistant", "tool", "user"]
    assert sent[2]["tool_call_id"] == "call_0"
    assert "did not answer" in sent[2]["content"]


async def test_unanswered_calls_can_raise(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")))
    agent = make_agent(tools=[delete_file], unanswered="raise")
    stopped = await agent.run("Delete a.txt")

    with pytest.raises(PendingRequestsError):
        await agent.run("Something else", history=stopped.messages)


async def test_waiting_calls_need_answers_or_a_prompt(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")))
    agent = make_agent(tools=[delete_file])
    stopped = await agent.run("Delete a.txt")

    with pytest.raises(ValueError, match="answers"):
        await agent.run(history=stopped.messages)


def test_unknown_unanswered_option_is_rejected(client: Client) -> None:
    from pyox_agents import Agent

    with pytest.raises(ValueError, match="unanswered"):
        Agent(client=client, model="m", unanswered="ignore")  # ty: ignore[invalid-argument-type]


# Hooks, history stores and the Responses API ------------------------------------------------------


async def test_after_run_sees_pending_requests(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    seen: list[list[str]] = []

    def remember(ctx: RunContext, result: RunFinished) -> None:
        seen.append([request.call_id for request in result.pending])

    fake_openai.queue(tool_calls_response(tool_call("delete_file", path="a.txt")))

    await make_agent(tools=[delete_file], hooks=[Hooks(after_run=remember)]).run("Delete a.txt")

    assert seen == [["call_0"]]


async def test_stopped_runs_resume_from_a_history_store(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    conversations: dict[str, list[Message]] = {}

    class MemoryHistory:
        async def load(self, conversation_id: str) -> list[Message]:
            return conversations.get(conversation_id, [])

        async def save(self, conversation_id: str, new_messages_json: str) -> None:
            conversations.setdefault(conversation_id, []).extend(json.loads(new_messages_json))

    fake_openai.queue(tool_calls_response(tool_call("book_flight", city="Oslo")), text_response("Booked"))
    agent = make_agent(tools=[book_flight], history_store=MemoryHistory())
    bookings = Bookings()

    await agent.run("Fly me to Oslo", conversation_id="c1", deps=bookings)
    result = await agent.run(conversation_id="c1", answers={"call_0": Answer("Oslo 9:00")}, deps=bookings)

    assert result.text == "Booked"
    assert bookings.booked == ["Oslo 9:00"]
    assert [message["role"] for message in conversations["c1"]] == ["user", "assistant", "tool", "tool", "assistant"]


async def test_stored_responses_resume_with_answers(fake_openai: FakeOpenAI, client: Client) -> None:
    from pyox_agents import Agent

    fake_openai.queue(
        responses_answer(tool_calls=(tool_call("delete_file", path="a.txt"),), response_id="resp_1"),
        responses_answer("Done", response_id="resp_2"),
    )
    agent = Agent(client=client, model="test-model", tools=[delete_file], api="responses", store=True)
    stopped = await agent.run("Delete a.txt")

    result = await agent.run(history=stopped.messages, answers={"call_0": Approve()})

    assert result.text == "Done"
    request = fake_openai.requests[1].body
    assert request["previous_response_id"] == "resp_1"
    assert request["input"] == [{"type": "function_call_output", "call_id": "call_0", "output": "deleted a.txt"}]


async def test_websocket_runs_resume_without_the_waiting_message(
    fake_openai: FakeOpenAI, websocket_client: Client
) -> None:
    from pyox_agents import Agent

    fake_openai.queue(
        responses_answer(tool_calls=(tool_call("delete_file", path="a.txt"),), response_id="resp_1"),
        responses_answer("Done", response_id="resp_2"),
    )
    agent = Agent(client=websocket_client, model="test-model", tools=[delete_file], api="responses")
    stopped = await agent.run("Delete a.txt")

    result = await agent.run(history=stopped.messages, answers={"call_0": Approve()})

    assert result.text == "Done"
    first, second = fake_openai.requests
    assert first.connection == second.connection
    # A new run sends its whole input, where the call has one result: the waiting message is left out.
    assert "previous_response_id" not in second.body
    assert [item.get("type", "message") for item in second.body["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]


# Frontend tools sent with a run (`frontend_tools=`) ----------------------------------------------


async def test_frontend_tools_can_come_with_each_run(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    location = {"name": "get_location", "description": "The user's location.", "parameters": {"type": "object"}}
    fake_openai.queue(tool_calls_response(tool_call("get_location")), text_response("Vigo it is"))
    agent = make_agent(tools=[delete_file])

    stopped = await agent.run("Where am I?", frontend_tools=[location])
    result = await agent.run(history=stopped.messages, answers={"call_0": Answer("Vigo")}, frontend_tools=[location])

    sent_tools = [schema["function"]["name"] for schema in fake_openai.requests[0].body["tools"]]
    assert sent_tools == ["delete_file", "get_location"]
    assert [(request.kind, request.tool_name) for request in stopped.pending] == [("frontend", "get_location")]
    assert result.text == "Vigo it is"
    assert len(agent.tools) == 1  # the frontend tool was only for those runs


def test_frontend_tools_need_a_name_of_their_own(make_agent: MakeAgent) -> None:
    agent = make_agent(tools=[delete_file])

    with pytest.raises(ValueError, match="agent's tools"):
        agent.execute("Hi", frontend_tools=[{"name": "delete_file"}])
    with pytest.raises(ValueError, match="needs a"):
        agent.execute("Hi", frontend_tools=[{"description": "No name"}])
