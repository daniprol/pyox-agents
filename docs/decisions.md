# Decisions API

OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions) answers questions about an input
with probabilities instead of text: is this statement true, which option fits, where on a scale. It is fast and
cheap (it only charges input tokens), which makes it a good fit for checks around an agent: triage, routing, guard
rails. It is in public beta, with one model, `gpt-6-luna`.

It is not a model turn, so it does not go through an agent: call `decide(...)` with the questions, by name.

```python
from pyox_agents import Choice, Predicate, Score, decide

decision = await decide(
    "The deploy failed twice and customers are seeing 500s. Can someone look now?",
    {
        "urgent": Predicate("Does this need attention right now?"),
        "team": Choice("Which team should handle it?", {"infra": "Deploys and outages.", "billing": "Payments."}),
        "severity": Score("How severe is the impact?", ["Cosmetic", "Degraded", "Outage"]),
    },
)

decision["urgent"].probability     # 0.97
decision["team"].choice            # "infra"
decision["team"].probabilities     # {"infra": 0.96, "billing": 0.04}
decision["severity"].score         # 1.8
```

All the questions go in one request, and each answer comes back under its question's name.

## Questions and answers

| Question | Asks | Answer |
|----------|------|--------|
| `Predicate(instructions)` | Is it true? | `PredicateAnswer`: `probability` (0 to 1) |
| `Choice(instructions, choices)` | Which option fits best? | `ChoiceAnswer`: `choice`, `confidence`, `probabilities` per option |
| `Score(instructions, levels)` | Where on a scale? | `ScoreAnswer`: `score`, `confidence`, `probabilities` per level |

- `choices` is a list of options (strings or booleans), or a dict from each option to a description.
- `levels` go from the lowest to the highest: a list of labels, or a dict from each label to a description.
  `score` is the probability-weighted average of the level indices, from 0, so it can fall between two levels.
- Any question can get a `Refusal` instead, when the model declines it. The other answers are still valid, so
  check the type before reading an answer:

```python
answer = decision["urgent"]
if isinstance(answer, PredicateAnswer) and answer.probability > 0.8:
    page_on_call()
```

`decision.answers` holds every answer by name, `decision.usage` the tokens (a [`Usage`](streaming.md)), and
`decision.request_id` OpenAI's id for the request, for support.

## Input

The input is any of:

- **Text**: `decide("The package arrived broken.", ...)`.
- **A prompt** with images, as for `agent.run`: `decide(["Is it broken?", image("photo.jpg")], ...)`. Files
  (such as PDFs) are not accepted by the API.
- **A history**, such as `result.messages`: the whole conversation with its tool calls, so you can ask about
  what happened in a run. The model's reasoning is left out (the API does not accept it).

```python
result = await agent.run("Cancel my subscription")
decision = await decide(result.messages, {"done": Predicate("Did the assistant cancel the subscription?")})
```

## Options

```python
await decide(
    input,
    questions,
    client=client,                # the same Client as your agents; Client() by default
    model="gpt-6-luna",           # the default
    safety_identifier="9f86d0",   # a hashed id of your user, for OpenAI's abuse checks
    prompt_cache_key="triage",    # groups requests that share a prefix, for the prompt cache
)
```

Errors are the same as for agents: `RateLimitError`, `AuthenticationError`, `ApiError`... (see
[Client and errors](client.md)). Retries and timeouts come from the `Client`.

## Recipes

**Ask for approval when a tool call looks risky.** A `before_tool_call` [hook](hooks.md) asks about each call, and
raises `ApprovalRequired` to make the risky ones wait for a person (see
[Human in the loop](human-in-the-loop.md)). A refusal counts as risky:

```python
async def risk_check(ctx: RunContext, call: ToolCall) -> None:
    decision = await decide(
        f"Tool call: {call['name']}({call['arguments']})",
        {"risky": Predicate("Could this tool call cause harm or lose data?")},
    )
    answer = decision["risky"]
    if not isinstance(answer, PredicateAnswer) or answer.probability >= 0.5:
        raise ApprovalRequired({"tool": call["name"]})


agent = Agent(model="gpt-5-mini", tools=[...], hooks=[Hooks(before_tool_call=risk_check)])
```

**Pick the agent for a request.** Ask a `Choice` before the run:

```python
agents = {"quick": Agent(model="gpt-5-mini"), "deep": Agent(model="gpt-5", prompt="Think it through.")}

decision = await decide(prompt, {"agent": Choice("How hard is this request?", {
    "quick": "Small talk or a simple question.",
    "deep": "Needs reasoning, several steps or care.",
})})
answer = decision["agent"]
agent = agents[answer.choice] if isinstance(answer, ChoiceAnswer) else agents["deep"]
result = await agent.run(prompt)
```

## Azure OpenAI

`decide` sends `POST /decisions` to the client's base URL, so `Client.azure(...)` works if your Azure resource
serves the Decisions API. Microsoft has not documented it yet; pass your deployment's name as `model`.
