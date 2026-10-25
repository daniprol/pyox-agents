from typing import Any

class RustTool:
    name: str
    description: str
    parameters: dict[str, Any]
    __pyox_native_tool__: object  # capsule the agent uses to call the Rust function directly
    def invoke(self, arguments: dict[str, Any]) -> str: ...

count_words: RustTool
most_common_words: RustTool
