# Benchmarks

How much time and memory does the agent library itself cost? These compare pyox-agents with
pydantic-ai, the OpenAI Agents SDK, Microsoft Agent Framework, LangGraph, LangChain, LlamaIndex and Agno on the
same tasks, against the same fake OpenAI server, which answers instantly. So the numbers are the overhead of
each library: called directly from Python, with each streaming mode, and behind a FastAPI streaming endpoint
under load.

--8<-- "benchmarks/README.md:results"

How it is measured, and how to run the benchmarks yourself, is in
[`benchmarks/`](https://github.com/daniprol/pyox-agents/tree/main/benchmarks).
