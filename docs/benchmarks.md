# Benchmarks

How much time and memory does the agent library itself cost? These compare pyox-agents with
[pydantic-ai](https://ai.pydantic.dev) on the same tasks, against the same fake OpenAI server, which answers
instantly. So the numbers are the overhead of each library.

--8<-- "benchmarks/README.md:results"

How it is measured, and how to run the benchmarks yourself, is in
[`benchmarks/`](https://github.com/daniprol/pyox-agents/tree/main/benchmarks).
