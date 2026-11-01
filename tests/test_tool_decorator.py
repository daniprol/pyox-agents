"""Building tools from Python functions."""

from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import pytest

from pyox_agents import Tool, tool


def test_schema_from_signature() -> None:
    @tool
    def search(
        query: Annotated[str, "What to look for"],
        limit: int = 10,
        score: float = 0.5,
        exact: bool = False,
        tags: list[str] | None = None,
        order: Literal["new", "old"] = "new",
        filters: dict[str, Any] | None = None,
        anything: Any = None,
        id_or_name: int | str = 0,
    ) -> str:
        """Search the documents."""
        return ""

    assert search.name == "search"
    assert search.description == "Search the documents."
    assert search.parameters == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for"},
            "limit": {"type": "integer"},
            "score": {"type": "number"},
            "exact": {"type": "boolean"},
            "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
            "order": {"type": "string", "enum": ["new", "old"]},
            "filters": {"anyOf": [{"type": "object"}, {"type": "null"}]},
            "anything": {},
            "id_or_name": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        },
        "required": ["query"],
    }


@dataclass
class Address:
    city: str
    street: Annotated[str, "Street and number"] = ""


class Contact(TypedDict):
    email: str


def test_dataclass_arguments_get_a_schema_and_are_built() -> None:
    @tool
    def ship(to: Address, stops: list[Address], contact: Contact | None = None) -> str:
        """Ship a parcel."""
        assert isinstance(to, Address)
        assert all(isinstance(stop, Address) for stop in stops)
        return f"{to.city} via {[stop.city for stop in stops]}, {contact}"

    address = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "street": {"type": "string", "description": "Street and number"}},
        "required": ["city"],
    }
    assert ship.parameters["properties"] == {
        "to": address,
        "stops": {"type": "array", "items": address},
        "contact": {
            "anyOf": [
                {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
                {"type": "null"},
            ]
        },
    }
    arguments = {"to": {"city": "Vigo"}, "stops": [{"city": "Lugo"}], "contact": {"email": "a@b.c"}}
    assert ship.invoke(arguments) == "Vigo via ['Lugo'], {'email': 'a@b.c'}"


def test_pydantic_arguments_by_duck_typing() -> None:
    pydantic = pytest.importorskip("pydantic")

    class Item(pydantic.BaseModel):
        name: str

    class Order(pydantic.BaseModel):
        items: list[Item]

    @tool
    def place(order: Order) -> str:
        """Place an order."""
        return ", ".join(item.name for item in order.items)

    assert place.parameters["properties"]["order"]["properties"]["items"]["items"] == {"$ref": "#/$defs/Item"}
    assert set(place.parameters["$defs"]) == {"Item"}  # moved to the top, where `$ref` finds it
    assert place.invoke({"order": {"items": [{"name": "tea"}]}}) == "tea"


def test_decorator_options() -> None:
    @tool(name="sum_numbers", description="Sum.", run_in_thread=True)
    def total(numbers: list) -> float:
        return sum(numbers)

    assert isinstance(total, Tool)
    assert (total.name, total.description, total.run_in_thread) == ("sum_numbers", "Sum.", True)
    assert total.parameters["properties"]["numbers"] == {"type": "array"}


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (lambda: "plain text", "plain text"),
        (lambda: {"total": 3.5}, '{"total": 3.5}'),
        (lambda: [1, 2], "[1, 2]"),
    ],
)
def test_results_are_text_or_json(function: Any, expected: str) -> None:
    assert tool(function).invoke({}) == expected


async def test_invoke_returns_a_coroutine_for_async_functions() -> None:
    async def double(text: str) -> str:
        return text * 2

    assert await tool(double).invoke({"text": "ab"}) == "abab"
