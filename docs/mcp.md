# MCP servers

[MCP](https://modelcontextprotocol.io) (Model Context Protocol) servers offer tools that live outside your
code: a local program, or a remote service. `McpTools` connects to one server and gives its tools to an agent as
ordinary tools, so approvals, hooks, subagents, tracing and web streams work on them as on any other tool.

```bash
pip install "pyox-agents[mcp]"   # adds the official `mcp` package
```

```python
from pyox_agents import Agent
from pyox_agents.mcp import McpTools

async with McpTools.http("https://mcp.deepwiki.com/mcp") as deepwiki:
    agent = Agent(model="gpt-5-mini", tools=deepwiki.tools)
    result = await agent.run("How does pydantic-ai run MCP servers?")
```

`async with` connects and lists the tools once. The connection stays open until the block ends; tools called
after that give the model an error. In a web app, open it once when the app starts (for example in a FastAPI
`lifespan`) and make agents inside it.

For [sync runs](sync.md) (`agent.run_sync`), open it with a plain `with McpTools.http(...) as deepwiki:` instead.
The connection then lives on pyox's own event loop thread, where sync runs call async tools.

When each user signs in to the server with their own token, open it for each request instead, and give its tools
to that run only with `tools=` (see [Tools for one run](tools.md#tools-for-one-run)):

```python
agent = Agent(model="gpt-5-mini")  # one agent for the app


async def answer(user, prompt):
    async with McpTools.http(GITHUB_URL, headers={"Authorization": f"Bearer {user.github_token}"}) as github:
        return await agent.run(prompt, tools=github.tools)
```

## Connecting

```python
# A remote server (Streamable HTTP), with headers sent on every request.
github = McpTools.http("https://api.githubcopilot.com/mcp/", headers={"Authorization": f"Bearer {token}"})

# A local server: pyox starts the command and talks to it over stdin/stdout.
files = McpTools.stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "./docs"])

async with github, files:
    agent = Agent(model="gpt-5", tools=[*github.tools, *files.tools])
```

A stdio server only gets a minimal environment: give it what it needs with `env={...}`.

For servers that sign in with OAuth, pass the `mcp` package's `OAuthClientProvider` as `auth=` to
`McpTools.http(...)`. `McpTools(server)` also takes anything `mcp.Client` connects to (a URL,
`mcp.StdioServerParameters`, a transport, or an `MCPServer` object in the same process, which is handy in tests),
and any other keyword argument goes to `mcp.Client`:

```python
McpTools("http://localhost:8000/mcp", read_timeout_seconds=60, mode="legacy")
```

`server.client` is the connected `mcp.Client`, for anything `McpTools` does not wrap.

## Choosing tools

Big servers have dozens of tools, and each one costs tokens and gives the model one more thing to get wrong.
Keep the ones you need, and name them apart when two servers clash:

```python
github = McpTools.http(GITHUB_URL, include={"search_issues", "create_issue"}, prefix="gh")
# tools: gh_search_issues, gh_create_issue
```

`exclude=` leaves tools out instead. Tool names are also cleaned up for OpenAI, which only accepts letters, digits,
`_` and `-` (so `files.read` becomes `files_read`). Each tool keeps the server's description and JSON schema, and
`tool.info` is the tool as the server listed it (its own name, annotations, output schema).

Another way to keep the main agent focused: give a big server to a [subagent](subagents.md).

```python
async with github:
    github_agent = Agent(model="gpt-5-mini", instructions="You work on GitHub.", tools=github.tools)
    agent = Agent(model="gpt-5", tools=[github_agent.as_tool("github", "Do anything on GitHub.")])
```

## Approvals

`requires_approval` says which tools wait for a person before each call (see
[Human in the loop](human-in-the-loop.md)):

```python
McpTools.http(GITHUB_URL, requires_approval="unless_read_only")
```

- `False` (default) or `True`: no tool, or every tool.
- `"unless_read_only"`: every tool the server did not mark as read-only. The marks come from the server, so treat
  them as a convenience, not as security.
- A set of tool names, as the server names them: `{"create_issue", "merge_pull_request"}`.
- A function getting each tool's `info` and returning what `@tool(requires_approval=...)` takes.

## Results

| The server returns | The model reads | Your app gets (`ToolFinished.metadata`) |
|---|---|---|
| text | the text | |
| an image | the image | |
| structured content | the text the server sent with it | `{"structured_content": ...}` |
| a resource or a link to one | its text, or a short note with its address | |
| audio | a short note (tool results cannot hold audio) | |
| an error | the error, as a tool error | |
| its own `_meta` | nothing | `{"meta": ...}` |

## When the server asks the user

A server may stop in the middle of a tool call to ask the user something (MCP calls it *elicitation*). It becomes
a question like a tool's `ctx.ask(...)`: the person answers it live, or the run stops and goes on once the answer
comes. The person sees `InputRequested.data`:

```python
{
    "type": "mcp_elicitation",
    "server": "rooms",
    "mode": "form",
    "message": "Which floor?",
    "schema": {"type": "object", "properties": {"floor": {"type": "integer"}}, "required": ["floor"]},
}
```

`schema` only has simple fields (text, numbers, true or false, choices), so a generic form can show it.
`Answer({"floor": 3})` accepts with those values, and `Deny("...")` declines: the call ends with that message as
its result.

```python
# Live: answer while the tool waits.
result = await agent.run("Book a room for Friday", on_request=lambda request: Answer({"floor": 3}))

# Or stop, and answer in a later run (another HTTP request, in a web app).
stopped = await agent.run("Book a room for Friday")
[request] = stopped.pending
result = await agent.run(history=stopped.messages, answers={request.call_id: Answer({"floor": 3})})
```

When the run goes on, the tool call runs again from the start and its question gets the stored answer. Servers
on MCP 2026-07-28 are built for this: they end the call, and the client calls again with the answer. Older servers
start the tool again from the beginning, so anything the tool did before asking happens twice.

Some questions send the user to a web page instead (`"mode": "url"`, with the `"url"` to open, for example to sign
in to another service). `Answer()` means the user is done there.

To answer some questions yourself, or to show them differently, subclass `McpTools` and override `elicit`:

```python
class MyServer(McpTools):
    async def elicit(self, ctx, params, *, key):
        if params.message == "Which floor?":
            return mcp_types.ElicitResult(action="accept", content={"floor": ctx.deps.home_floor})
        return await super().elicit(ctx, params, key=key)
```

`make_tool` (how a listed tool becomes an agent tool) and `convert_result` (what the model and the app get from a
result) can be overridden the same way.

## Prompts and resources

These are for your app, not for the model:

```python
messages = await server.prompt("review", day="friday")  # messages, e.g. for agent.run(history=...)
rules = await server.resource("rooms://rules")  # text, images and files, to put in a prompt
```

## Tracing

With [tracing](telemetry.md) on, each MCP call sends the trace it belongs to (`traceparent`), so a server that is
traced too shows its work under the agent's `execute_tool` span.

## MCP versions

The `mcp` package talks to servers on every MCP version, from 2024-11-05 to 2026-07-28, and picks the newest both
sides know. The tests cover 2026-07-28 and 2025-11-25 over stdio, Streamable HTTP and in-process servers.
Servers that still use the old HTTP+SSE transport (deprecated since 2025) can be reached with the `mcp` package's
`sse_client`: `McpTools(sse_client(url))`. Sampling, roots and logging, which MCP 2026-07-28 deprecated, are left
out.
