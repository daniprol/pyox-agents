from typing import Any

class RustTool:
    name: str
    description: str
    parameters: dict[str, Any]
    def invoke(self, arguments: dict[str, Any]) -> str: ...

count_words: RustTool
most_common_words: RustTool
