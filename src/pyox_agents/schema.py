"""JSON schemas from Python types, and Python values back from JSON.

Tool arguments and structured output (`Agent(output_type=...)`) both use it:

    json_schema(Ticket)             # the JSON schema of a dataclass, TypedDict, pydantic model...
    strict_schema(schema)           # the same schema in OpenAI's strict mode
    from_json(Ticket, {"team": ...})  # a `Ticket` again

Supported: `str`, `int`, `float`, `bool`, `None`, `list[...]`, `dict[...]`,
`Literal[...]`, unions (`X | None`), `Annotated[X, "description"]`,
dataclasses, `TypedDict`s, and pydantic models (by duck typing: anything with
`model_json_schema()` and `model_validate()`; pydantic is never imported).
Anything else accepts any JSON value.
"""

import dataclasses
import json
import re
import types
from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints, is_typeddict

JsonSchema = dict[str, Any]

_SIMPLE_TYPES: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean", type(None): "null"}


def json_schema(annotation: Any) -> JsonSchema:
    """The JSON schema for a Python type annotation."""
    schema = _schema(annotation)
    # Pydantic puts nested models in `$defs` and points at them with `#/$defs/Name`,
    # which only works at the top of the schema: move them all there.
    definitions: JsonSchema = {}
    schema = _take_definitions(schema, definitions)
    return {**schema, "$defs": definitions} if definitions else schema


def _take_definitions(schema: Any, definitions: JsonSchema) -> Any:
    """`schema` without its `$defs`, which are added to `definitions`."""
    if isinstance(schema, list):
        return [_take_definitions(item, definitions) for item in schema]
    if not isinstance(schema, dict):
        return schema
    for name, definition in schema.get("$defs", {}).items():
        definitions[name] = _take_definitions(definition, definitions)
    return {key: _take_definitions(value, definitions) for key, value in schema.items() if key != "$defs"}


def _schema(annotation: Any) -> JsonSchema:
    origin, args = get_origin(annotation), get_args(annotation)

    if annotation in _SIMPLE_TYPES:
        return {"type": _SIMPLE_TYPES[annotation]}
    if origin is Annotated:
        schema = _schema(args[0])
        descriptions = [extra for extra in args[1:] if isinstance(extra, str)]
        return {**schema, "description": descriptions[0]} if descriptions else schema
    if origin is Literal:
        value_types = {_SIMPLE_TYPES.get(type(value)) for value in args}
        if len(value_types) == 1 and None not in value_types:
            return {"type": value_types.pop(), "enum": list(args)}
        return {"enum": list(args)}
    if origin in (Union, types.UnionType):
        options = [_schema(arg) for arg in args]
        return options[0] if len(options) == 1 else {"anyOf": options}
    if annotation is list or origin is list:
        return {"type": "array", "items": _schema(args[0])} if args else {"type": "array"}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    if _is_pydantic_model(annotation):
        return annotation.model_json_schema()
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        fields = dataclasses.fields(annotation)
        required = [field.name for field in fields if not _has_default(field)]
        return _object_schema(annotation, [field.name for field in fields], required)
    if is_typeddict(annotation):
        return _object_schema(annotation, list(get_type_hints(annotation)), sorted(annotation.__required_keys__))
    return {}  # Any or unknown: accept any JSON value


def _object_schema(cls: type, names: list[str], required: list[str]) -> JsonSchema:
    type_hints = get_type_hints(cls, include_extras=True)
    properties = {name: _schema(type_hints.get(name, Any)) for name in names}
    return {"type": "object", "properties": properties, "required": required}


def _has_default(field: dataclasses.Field[Any]) -> bool:
    return field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING


def _is_pydantic_model(annotation: Any) -> bool:
    return (
        isinstance(annotation, type)
        and hasattr(annotation, "model_json_schema")
        and hasattr(annotation, "model_validate")
    )


def strict_schema(schema: JsonSchema) -> JsonSchema:
    """A copy of `schema` in OpenAI's strict mode, which guarantees the model's answer matches it.

    Strict mode needs every object to list all its properties as required and to allow no
    others (`additionalProperties: false`). So a field with a default must be given by the
    model too; a field that may be missing should be typed `X | None`, which allows `null`.
    """
    strict = dict(schema)
    if strict.get("type") == "object" and "properties" in strict:
        strict["properties"] = {name: strict_schema(value) for name, value in strict["properties"].items()}
        strict["required"] = list(strict["properties"])
        strict["additionalProperties"] = False
    if isinstance(strict.get("items"), dict):
        strict["items"] = strict_schema(strict["items"])
    for key in ("anyOf", "allOf"):
        if key in strict:
            strict[key] = [strict_schema(option) for option in strict[key]]
    for key in ("$defs", "definitions"):  # pydantic puts nested models here
        if key in strict:
            strict[key] = {name: strict_schema(value) for name, value in strict[key].items()}
    if strict.get("default", ...) is None:
        del strict["default"]  # strict mode rejects `"default": null` (pydantic writes it for `X | None = None`)
    return strict


def from_json(annotation: Any, value: Any) -> Any:
    """`value` (parsed JSON) as the type `annotation` describes: dataclasses and pydantic models
    are built, also inside lists and unions; everything else is returned as it is."""
    origin, args = get_origin(annotation), get_args(annotation)
    if value is None:
        return None
    if origin is Annotated:
        return from_json(args[0], value)
    if origin in (Union, types.UnionType):
        options = [arg for arg in args if needs_conversion(arg)]
        return from_json(options[0], value) if len(options) == 1 else value
    if origin is list and args and isinstance(value, list):
        return [from_json(args[0], item) for item in value]
    if _is_pydantic_model(annotation):
        return annotation.model_validate(value)
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type) and isinstance(value, dict):
        type_hints = get_type_hints(annotation, include_extras=True)
        fields = {field.name for field in dataclasses.fields(annotation)}
        arguments = {name: from_json(type_hints.get(name), item) for name, item in value.items() if name in fields}
        return annotation(**arguments)
    return value


def type_problems(annotation: Any, value: Any, path: str) -> list[str]:
    """What is wrong with `value` (parsed JSON) as a `annotation`, one line per problem,
    each starting with where it is (`path`, like `trip.days`). Empty when it fits.

    Checks the types `json_schema` supports. Pydantic models are left to pydantic
    (see `validation_problems`); anything else accepts any value."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return type_problems(args[0], value, path)
    if annotation in _TYPE_NAMES:
        return [] if _is_simple_type(annotation, value) else [_wrong(path, _TYPE_NAMES[annotation], value)]
    if origin is Literal:
        return [] if value in args else [_wrong(path, "one of " + ", ".join(json.dumps(arg) for arg in args), value)]
    if origin in (Union, types.UnionType):
        problems_per_option = [type_problems(arg, value, path) for arg in args]
        if any(not problems for problems in problems_per_option):
            return []
        options = [arg for arg in args if arg is not type(None)]
        return (
            problems_per_option[args.index(options[0])] if len(options) == 1 else [_wrong(path, "another type", value)]
        )
    if annotation is list or origin is list:
        if not isinstance(value, list):
            return [_wrong(path, "a list", value)]
        return [
            problem
            for index, item in enumerate(value)
            for problem in type_problems(args[0] if args else Any, item, f"{path}[{index}]")
        ]
    if annotation is dict or origin is dict:
        return [] if isinstance(value, dict) else [_wrong(path, "an object", value)]
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type) and not _is_pydantic_model(annotation):
        fields = dataclasses.fields(annotation)
        required = [field.name for field in fields if not _has_default(field)]
        return _object_problems(annotation, [field.name for field in fields], required, value, path)
    if is_typeddict(annotation):
        names = list(get_type_hints(annotation))
        return _object_problems(annotation, names, sorted(annotation.__required_keys__), value, path)
    return []


_TYPE_NAMES: dict[Any, str] = {
    str: "a string",
    int: "an integer",
    float: "a number",
    bool: "true or false",
    type(None): "null",
}


def _is_simple_type(annotation: Any, value: Any) -> bool:
    if annotation is float:  # JSON has one number type: 3 is a fine float
        return isinstance(value, int | float) and not isinstance(value, bool)
    if annotation is int:  # `True` is an `int` in Python, but not in JSON
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, annotation)


def _object_problems(cls: type, names: list[str], required: list[str], value: Any, path: str) -> list[str]:
    if not isinstance(value, dict):
        return [_wrong(path, "an object", value)]
    type_hints = get_type_hints(cls, include_extras=True)
    problems = [f"{path}.{name}: missing" for name in required if name not in value]
    for name in names:
        if name in value:
            problems.extend(type_problems(type_hints.get(name, Any), value[name], f"{path}.{name}"))
    return problems


def _wrong(path: str, expected: str, value: Any) -> str:
    shown = json.dumps(value)
    return f"{path}: expected {expected}, got {shown if len(shown) <= 60 else shown[:57] + '...'}"


def validation_problems(error: Exception, path: str) -> list[str]:
    """`error` (raised while building an argument) as problem lines, like `type_problems`.
    A pydantic `ValidationError` gives one short line per error, without its links."""
    errors = getattr(error, "errors", None)
    if not callable(errors):
        return [f"{path}: {error}"]
    problems = []
    for details in errors():
        where = ".".join([path, *(str(part) for part in details.get("loc", ()))])
        shown = json.dumps(details.get("input"), default=str)
        given = f" (got {shown})" if "input" in details and len(shown) <= 60 else ""
        problems.append(f"{where}: {details.get('msg', 'invalid')}{given}")
    return problems


def needs_conversion(annotation: Any) -> bool:
    """Whether `from_json` would build something (a dataclass or a pydantic model) from JSON."""
    if _is_pydantic_model(annotation) or (dataclasses.is_dataclass(annotation) and isinstance(annotation, type)):
        return True
    return any(needs_conversion(arg) for arg in get_args(annotation) if not isinstance(arg, str))


def output_format(output_type: Any) -> tuple[JsonSchema, Callable[[str], Any]]:
    """For `Agent(output_type=...)`: the `{"name", "schema", "strict"}` object the API gets,
    and the function turning the answer's text into an `output_type` value.

    `output_type` is a dataclass, a `TypedDict`, a pydantic model, or a JSON schema (a dict:
    the answer is then a dict too)."""
    if isinstance(output_type, Mapping):
        schema, name = dict(output_type), output_type.get("title", "output")
    else:
        schema, name = json_schema(output_type), getattr(output_type, "__name__", "output")
    if schema.get("type") != "object":
        raise TypeError(
            f"output_type must describe a JSON object (a dataclass, TypedDict, pydantic model or object schema), "
            f"not {output_type!r}"
        )

    def parse(text: str) -> Any:
        value = json.loads(text)
        return value if isinstance(output_type, Mapping) else from_json(output_type, value)

    # The API accepts names of letters, digits, `_` and `-`, up to 64 characters.
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))[:64] or "output"
    return {"name": name, "schema": strict_schema(schema), "strict": True}, parse
