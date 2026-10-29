"""The agents we measure, all doing the same job.

Each `make_*` function returns `stream(prompt)`: an async iterator over the pieces
of answer text, as they arrive. Both benchmarks use it: `measure.py` calls it
directly, `server.py` sends the pieces to an HTTP client.
Libraries are imported inside the functions, so a process only loads the one it measures.
"""

from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyox_agents.agent import TextChunks

StreamText = Callable[[str], AsyncIterator[str]]

MODEL = "gpt-5-mini"  # any name works: the fake server ignores it
PROMPT = "You are a helpful assistant."


def get_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"Sunny, 21 degrees in {city}"


def make_pyox_agent(base_url: str, text_chunks: "TextChunks" = "delta") -> StreamText:
    from pyox_agents import Agent, Client, TextDelta

    agent = Agent(
        client=Client(api_key="not-needed", base_url=base_url),
        model=MODEL,
        prompt=PROMPT,
        tools=[get_weather],
        text_chunks=text_chunks,
    )

    async def stream(prompt: str) -> AsyncIterator[str]:
        async for event in agent.execute(prompt):
            if isinstance(event, TextDelta):
                yield event.text

    return stream


def make_pydantic_ai_agent(base_url: str) -> StreamText:
    from pydantic_ai import Agent, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    agent = Agent(
        OpenAIChatModel(MODEL, provider=OpenAIProvider(api_key="not-needed", base_url=base_url)),
        instructions=PROMPT,
        tools=[get_weather],
    )

    async def stream(prompt: str) -> AsyncIterator[str]:
        async with agent.run_stream_events(prompt) as events:
            async for event in events:
                # Text arrives as the start of a text part, then as deltas of that part.
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    yield event.part.content
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    yield event.delta.content_delta

    return stream


def make_openai_agents_agent(base_url: str) -> StreamText:
    from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
    from openai import AsyncOpenAI
    from openai.types.responses import ResponseTextDeltaEvent

    set_tracing_disabled(True)  # on by default: it would send traces to OpenAI
    agent = Agent(
        name="assistant",
        instructions=PROMPT,
        # The default model uses the Responses API; the fake server speaks Chat Completions.
        model=OpenAIChatCompletionsModel(MODEL, openai_client=AsyncOpenAI(api_key="not-needed", base_url=base_url)),
        tools=[function_tool(get_weather)],
    )

    async def stream(prompt: str) -> AsyncIterator[str]:
        result = Runner.run_streamed(agent, prompt)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta

    return stream


def make_agent_framework_agent(base_url: str) -> StreamText:
    from agent_framework import Agent
    from agent_framework.openai import OpenAIChatCompletionClient  # `OpenAIChatClient` is the Responses API

    agent = Agent(
        client=OpenAIChatCompletionClient(model=MODEL, api_key="not-needed", base_url=base_url),
        instructions=PROMPT,
        tools=[get_weather],
    )

    async def stream(prompt: str) -> AsyncIterator[str]:
        async for update in agent.run(prompt, stream=True):
            if update.text:
                yield update.text

    return stream


def make_langgraph_agent(base_url: str) -> StreamText:
    """The usual LangGraph agent: a model node and a tool node in a loop."""
    from langchain_openai import ChatOpenAI
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    model = ChatOpenAI(model=MODEL, api_key="not-needed", base_url=base_url).bind_tools([get_weather])

    async def call_model(state: MessagesState) -> dict:
        return {"messages": [await model.ainvoke([("system", PROMPT), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode([get_weather]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)  # to "tools" if the model asked for tools, else the end
    graph.add_edge("tools", "model")
    return stream_langgraph(graph.compile())


def make_langchain_agent(base_url: str) -> StreamText:
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=MODEL, api_key="not-needed", base_url=base_url)
    return stream_langgraph(create_agent(model, tools=[get_weather], system_prompt=PROMPT))


def stream_langgraph(graph) -> StreamText:
    """LangChain agents are LangGraph graphs: both stream the same way."""

    async def stream(prompt: str) -> AsyncIterator[str]:
        # "messages" mode streams the model's tokens; tool-call chunks have no text content.
        async for chunk, _ in graph.astream({"messages": [("user", prompt)]}, stream_mode="messages"):
            if chunk.type == "AIMessageChunk" and isinstance(chunk.content, str) and chunk.content:
                yield chunk.content

    return stream


def make_llama_index_agent(base_url: str) -> StreamText:
    from llama_index.core.agent.workflow import AgentStream, FunctionAgent
    from llama_index.llms.openai import OpenAI

    agent = FunctionAgent(
        llm=OpenAI(model=MODEL, api_key="not-needed", api_base=base_url),
        tools=[get_weather],
        system_prompt=PROMPT,
    )

    async def stream(prompt: str) -> AsyncIterator[str]:
        handler = agent.run(prompt)
        async for event in handler.stream_events():
            if isinstance(event, AgentStream) and event.delta:
                yield event.delta
        await handler  # let the run finish and raise its errors, if any

    return stream


def make_agno_agent(base_url: str) -> StreamText:
    from agno.agent import Agent, RunEvent
    from agno.models.openai import OpenAIChat

    agent = Agent(
        model=OpenAIChat(id=MODEL, api_key="not-needed", base_url=base_url),
        instructions=PROMPT,
        tools=[get_weather],
        telemetry=False,  # on by default: it would send a request to Agno after every run
    )

    async def stream(prompt: str) -> AsyncIterator[str]:
        async for event in agent.arun(prompt, stream=True):
            if event.event == RunEvent.run_content and isinstance(event.content, str) and event.content:
                yield event.content

    return stream


# Comparison 1: the same agent in each library.
LIBRARIES: dict[str, Callable[[str], StreamText]] = {
    "pyox-agents": make_pyox_agent,
    "pydantic-ai": make_pydantic_ai_agent,
    "openai-agents": make_openai_agents_agent,
    "agent-framework": make_agent_framework_agent,
    "langgraph": make_langgraph_agent,
    "langchain": make_langchain_agent,
    "llama-index": make_llama_index_agent,
    "agno": make_agno_agent,
}
# The installed package of each library, to report its version.
PACKAGES = {
    "pyox-agents": "pyox-agents",
    "pydantic-ai": "pydantic-ai-slim",
    "openai-agents": "openai-agents",
    "agent-framework": "agent-framework-core",
    "langgraph": "langgraph",
    "langchain": "langchain",
    "llama-index": "llama-index-core",
    "agno": "agno",
}

# Comparison 2: pyox-agents with each way of grouping streamed text (`text_chunks`).
TEXT_CHUNKS_MODES: list["TextChunks"] = ["delta", 10, "paragraph", "message"]
TEXT_CHUNKS: dict[str, Callable[[str], StreamText]] = {
    f"text_chunks={mode!r}": partial(make_pyox_agent, text_chunks=mode) for mode in TEXT_CHUNKS_MODES
}

AGENTS = LIBRARIES | TEXT_CHUNKS
