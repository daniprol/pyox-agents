# Client and errors

## Client

The client holds the API key, the server address and the connection pool. Without one, the agent creates a
default client from the environment (`OPENAI_API_KEY`, `OPENAI_BASE_URL`).

```python
from pyox_agents import Agent, Client

client = Client(
    api_key="sk-...",  # default: $OPENAI_API_KEY
    base_url="https://api.openai.com/v1",  # default: $OPENAI_BASE_URL or OpenAI
    headers={"X-Application": "support"},
    connect_timeout_seconds=10,
    request_timeout_seconds=600,  # covers the whole streamed answer
    max_retries=2,  # resend requests that failed before any response
)
agent = Agent(client=client, model="gpt-5-mini")
```

Share one `Client` between agents to reuse its connections.

A request that fails before any response arrives (for example on a pooled connection the server had already
closed) is sent again, up to `max_retries` times. Once the answer starts streaming it is never retried.

### Other OpenAI-compatible servers

Point `base_url` at any server that speaks the Chat Completions API:

```python
client = Client(base_url="http://localhost:11434/v1", api_key="ollama")  # e.g. Ollama
agent = Agent(client=client, model="llama3.2")
```

### Azure OpenAI

`Client.azure(...)` talks to Azure OpenAI (Microsoft Foundry) through its
[v1 API](https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle), which speaks the same protocol as
OpenAI. There is no `api-version` to pick, and the agent's `model` is the name of your deployment:

```python
client = Client.azure(
    "https://my-resource.openai.azure.com",  # default: $AZURE_OPENAI_ENDPOINT
    api_key="...",  # default: $AZURE_OPENAI_API_KEY
)
agent = Agent(client=client, model="my-gpt-5-mini-deployment", api="responses")
```

It is a normal `Client` whose `base_url` is `https://my-resource.openai.azure.com/openai/v1`, so it takes the
same options (`headers`, timeouts, `max_retries`). `Client(base_url=".../openai/v1", api_key=...)` does the same.
Both APIs work (`api="chat"` and `api="responses"`). For Microsoft Entra ID, pass a token as `api_key`; tokens
expire, so create a new client with a fresh one.

## Errors

All exceptions inherit from `AgentError`, so one `except` catches them all:

| Exception                | Raised when                                                       |
|--------------------------|-------------------------------------------------------------------|
| `ApiError`               | the API answered with an error                                    |
| `AuthenticationError`    | the API key is missing or wrong (an `ApiError`)                   |
| `RateLimitError`         | the API is rate limiting you (an `ApiError`)                      |
| `RequestTimeoutError`    | a request took longer than its timeout                            |
| `NetworkError`           | the server could not be reached                                   |
| `InvalidResponseError`   | the API sent something that could not be parsed                   |
| `MaxTurnsExceededError`  | the run made more than `max_turns` model requests (20 by default) |
| `HistoryError`           | a history store returned something invalid                        |
| `HookError`              | a Rust hook failed, or a hook returned something invalid          |

```python
from pyox_agents import AgentError, RateLimitError

try:
    result = await agent.run("Hi")
except RateLimitError:
    ...  # wait and try again
except AgentError as error:
    print(f"The run failed: {error}")
```

Errors inside tools don't raise: the model gets the error message and the run continues
(see [Tools](tools.md#python-tools)).
