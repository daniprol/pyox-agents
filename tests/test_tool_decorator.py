"""Building tools from Python functions."""

from typing import Annotated, Any, Literal

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
            "tags": {"type": "array", "items": {"type": "string"}},
            "order": {"enum": ["new", "old"]},
            "filters": {"type": "object"},
            "anything": {},
            "id_or_name": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        },
        "required": ["query"],
    }


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
async def test_results_are_text_or_json(function: Any, expected: str) -> None:
    assert await tool(function).invoke({}) == expected


async def test_invoke_awaits_async_functions_and_runs_threads() -> None:
    async def double(text: str) -> str:
        return text * 2

    def blocking_upper(text: str) -> str:
        return text.upper()

    assert await tool(double).invoke({"text": "ab"}) == "abab"
    assert await tool(run_in_thread=True)(blocking_upper).invoke({"text": "hi"}) == "HI"
