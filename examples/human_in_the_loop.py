"""Tools that wait for a person: one asks a question, one needs approval.

OPENAI_API_KEY=... uv run examples/human_in_the_loop.py
"""

import asyncio

from pyox_agents import Agent, Answer, Approve, Deny, InputRequested, Reply, RunContext, tool


@tool
async def book_table(ctx: RunContext, restaurant: str) -> str:
    """Book a table at a restaurant."""
    time = await ctx.ask({"question": f"What time at {restaurant}?", "options": ["20:00", "21:30"]})
    return f"Booked {restaurant} at {time.value}"


@tool(requires_approval=True)
def delete_file(path: str) -> str:
    """Delete a file."""
    return f"Deleted {path}"  # a real tool would delete it here


agent = Agent(model="gpt-5-mini", prompt="Be brief.", tools=[book_table, delete_file])


def ask_in_the_terminal(request: InputRequested) -> Reply:
    if request.kind == "approval":
        allowed = input(f"Allow {request.tool_name} {request.arguments}? [y/n] ") == "y"
        return Approve() if allowed else Deny()
    return Answer(input(f"{request.data['question']} {request.data['options']} "))


async def main() -> None:
    # Stop and resume: the run ends with `pending` requests and a later run sends the answers.
    # In a web app, the two runs would be two HTTP requests.
    result = await agent.run("Book a table at Casa Marcelo")
    while result.pending:
        answers = {request.call_id: ask_in_the_terminal(request) for request in result.pending}
        result = await agent.run(history=result.messages, answers=answers)
    print(result.text)

    # Live: the handler replies while the tool waits.
    result = await agent.run("Delete old.log", on_request=ask_in_the_terminal)
    print(result.text)


asyncio.run(main())
