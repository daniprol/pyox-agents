# Responses API

Agents use OpenAI's Chat Completions API by default. Pass `api="responses"` to use the
[Responses API](https://platform.openai.com/docs/api-reference/responses) instead. Everything else (tools,
streaming, history, hooks) works the same way.

```python
agent = Agent(model="gpt-5-mini", api="responses")
```

The history format does not change either, so a conversation can move between the two APIs.

## Encrypted reasoning

Reasoning models think before they answer. The Responses API can return that reasoning encrypted: you cannot read
it, but sending it back lets the model build on it in the next request. pyox-agents asks for it
(`include: ["reasoning.encrypted_content"]`), keeps it in the assistant message and sends it back unchanged:

```python
{
    "role": "assistant",
    "content": "It is 5.",
    "reasoning": [{"id": "rs_1", "summary": ["Adding 2 and 3."], "encrypted_content": "gAAAA..."}],
    "response_id": "resp_1",
    "model": "gpt-5-mini",
}
```

Because the reasoning goes back byte for byte, the start of each request stays the same and OpenAI's prompt cache
keeps working. `RunFinished.usage.cached_input_tokens` shows how many input tokens came from the cache.

Reasoning is only sent back to the model that wrote it (the message's `model`); for any other model it is left out.
`summary` holds the readable summaries, if you ask for them:

```python
agent = Agent(
    model="gpt-5-mini",
    api="responses",
    model_parameters={"reasoning": {"effort": "low", "summary": "auto"}},
)
```

Models that do not reason (such as gpt-4.1) reject the encrypted reasoning request. Turn it off for them:

```python
agent = Agent(model="gpt-4.1", api="responses", encrypted_reasoning=False)
```

## Stored responses: `store` and `previous_response_id`

By default every request is stateless (`store: false`): OpenAI keeps nothing, and each request sends the whole
conversation. This works with zero data retention, and your history stays the only copy.

With `store=True`, OpenAI keeps each response, and a request can continue from the last one
(`previous_response_id`) and send only the new messages. pyox-agents does this for you:

- **Inside a run**, after the first answer, each request sends only the new tool results.
- **Across runs**, a history whose last answer has a `response_id` sends only the messages after it. This works
  with `history=` and with history stores.
- **With only an id**, a run can continue a stored conversation without any history:

```python
agent = Agent(model="gpt-5-mini", api="responses", store=True)

first = await agent.run("I am Dani")
second = await agent.run("What is my name?", previous_response_id=first.response_id)
```

If OpenAI no longer has the response (it was deleted, or is too old), the request is sent again once with the whole
history, when the run has it. With only a `previous_response_id`, the error is raised.

If a `before_model_request` hook changes the messages, OpenAI's stored copy no longer matches them, so that
request sends all of them instead of continuing from the stored response.

## OpenAI conversations: `openai_conversation`

OpenAI's [Conversations API](https://platform.openai.com/docs/api-reference/conversations) keeps a whole
conversation on OpenAI's side, with no 30-day limit like stored responses. Each request names the conversation,
OpenAI puts its items in front of the request's input and adds the new input and the answer to it.

pyox-agents does not create or delete conversations: do that with the OpenAI SDK (or a plain HTTP request), then
pass the id to each run:

```python
from openai import AsyncOpenAI

conversation = await AsyncOpenAI().conversations.create(metadata={"user": "dani"})

agent = Agent(model="gpt-5-mini", api="responses")
await agent.run("I am Dani", openai_conversation=conversation.id)
result = await agent.run("What is my name?", openai_conversation=conversation.id)
```

Each request sends only what the conversation does not have yet:

- **Inside a run**, after the first answer, only the new tool results.
- **Across runs**, each answer added to the conversation is marked in the history (`"openai_conversation":
  "conv_..."`), and the next run sends only the messages after the last marked one. A history without marks (or
  with another conversation's) is sent whole, so it starts the new conversation with the same context. A run with
  no history sends only its prompt.

Requests with a conversation are always stored (`store: true`, even if the agent has `store=False`): otherwise
OpenAI would not add them to it. They never use `previous_response_id`, which the API does not accept next to a
conversation, so `execute(previous_response_id=..., openai_conversation=...)` raises `ValueError`.

A history store works as usual: it keeps your own copy of the conversation, for example to show it in your app.
`conversation_id=` names that copy; `openai_conversation=` names OpenAI's.

```python
await agent.run("Hi", conversation_id="user-42", openai_conversation=conversation.id)
```

Things to know:

- `before_model_request` hooks cannot change what the conversation already holds: only the messages sent with the
  request.
- OpenAI locks a conversation while a response is being made, so two runs on the same conversation at the same
  time fail.
- Over WebSocket, the conversation is sent the same way. OpenAI's SDK accepts it there, but the WebSocket guide
  does not mention it, so it is not tested against OpenAI yet.
- Azure OpenAI documents the Conversations API in its v1 API, but it may not be available in every region yet.

## WebSocket mode

`Client(transport="websocket")` sends Responses API requests over WebSocket connections. A run keeps one
connection for all its turns, and each turn after the first sends only the new tool results on top of the
previous response, also with `store=False`: the server keeps the latest response of each connection in memory.
OpenAI reports up to about 40% faster runs with many tool calls. See [Client](client.md#websocket-mode).

## Other settings

Anything else goes in `model_parameters` and is passed through as is, for example `reasoning`, `truncation`,
`service_tier`, `prompt_cache_key`, `metadata` or `max_output_tokens`. Your `include` list is kept, and the
encrypted reasoning is added to it. The fields pyox-agents controls (`model`, `input`, `instructions`, `stream`,
`store`, `tools`, `previous_response_id`) always win.

Built-in tools of the Responses API (web search, file search, code interpreter) are not supported yet.

## Azure OpenAI

Azure OpenAI (Microsoft Foundry) supports the Responses API through its v1 API. See
[Client: Azure OpenAI](client.md#azure-openai).
