import asyncio
from typing import Annotated, Any, Literal

from pyox_agents import Tool, tool


def test_schema_from_signature() -> None:
    @tool
    def search(
        query: Annotated[str, "What to look for"],
        limit: int = 10,
        tags: list[str] | None = None,
        order: Literal["new", "old"] = "new",
        filters: dict[str, Any] | None = None,
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
            "tags": {"type": "array", "items": {"type": "string"}},
            "order": {"enum": ["new", "old"]},
            "filters": {"type": "object"},
        },
        "required": ["query"],
    }


def test_decorator_options_and_invoke() -> None:
    @tool(name="sum_numbers", description="Sum.", run_in_thread=True)
    def total(numbers: list[float]) -> dict[str, float]:
        return {"total": sum(numbers)}

    async def lookup(key: str) -> str:
        return key * 2

    assert isinstance(total, Tool)
    assert total.name == "sum_numbers"
    assert asyncio.run(total.invoke({"numbers": [1, 2.5]})) == '{"total": 3.5}'
    assert asyncio.run(tool(lookup).invoke({"key": "ab"})) == "abab"
