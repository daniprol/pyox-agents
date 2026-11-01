# Structured output

Give the agent an `output_type` and its answer is JSON matching that type, read back as a Python value in
`RunFinished.output`:

```python
--8<-- "examples/structured_output.py"
```

`output_type` can be:

| `output_type`                     | `RunFinished.output`         |
|-----------------------------------|------------------------------|
| a dataclass                       | an instance of it            |
| a pydantic model                  | an instance of it            |
| a `TypedDict`                     | a dict                       |
| a JSON schema (a dict)            | a dict                       |

Fields can use the same types as [tool parameters](tools.md#python-tools): nested dataclasses, lists,
`Literal`s, `X | None`, `Annotated[str, "description"]`... Pydantic models are used by duck typing (their
`model_json_schema()` and `model_validate()`), so pydantic is not a dependency.

## Strict mode

The schema is sent in OpenAI's strict mode (`response_format` with Chat Completions, `text.format` with the
Responses API), which guarantees the answer matches it. Strict mode has rules of its own, which pyox-agents
applies for you:

- Every field is required, even those with a default. A field the model may leave empty should be typed
  `X | None`: it then answers `null`.
- No extra fields are allowed (`additionalProperties: false`).
- The top level must be an object: a dataclass, `TypedDict`, pydantic model or object schema, not `list[str]`.

Free-form fields (`Any`, `dict[str, Any]`) are not allowed in strict mode: the API rejects the schema.

## When there is no output

`output` is `None` when the run did not end with an answer: it was cut by the token limit, the model refused, a
tool call waits for a person, or `run.stop()` was called (see [`stop_reason`](streaming.md#how-a-run-ends)).
`text` still has whatever the model wrote. An answer that does not match the type (rare in strict mode) raises
`InvalidResponseError`, with the conversation in `error.messages`.

`agent.run(...)` is typed: `Agent(output_type=Ticket)` gives a `RunFinished[Ticket]`, so `result.output` is a
`Ticket | None` to your type checker.
