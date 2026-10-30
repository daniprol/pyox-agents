"""An agent behind a web server, for an AI SDK (`useChat`) frontend and for an AG-UI (CopilotKit) one.

OPENAI_API_KEY=... uv run uvicorn examples.web_server:app --reload

In both, the conversation lives in the browser and comes with each request.
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from pyox_agents import Agent, RunContext, ag_ui, ai_sdk, tool


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
app = FastAPI()


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    """For `useChat()` from the AI SDK (`DefaultChatTransport({api: "/api/chat"})`)."""
    return StreamingResponse(ai_sdk.stream(agent, await request.body()), headers=ai_sdk.HEADERS)


@app.post("/api/agent")
async def run_agent(request: Request) -> StreamingResponse:
    """For an AG-UI client (`new HttpAgent({url: "/api/agent"})`, CopilotKit)."""
    return StreamingResponse(ag_ui.stream(agent, await request.body()), headers=ag_ui.HEADERS)
