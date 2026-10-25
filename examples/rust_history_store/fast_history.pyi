from pyox_agents import Message

class RustHistoryStore:
    __pyox_history_store__: object  # capsule the agent uses to call the Rust functions directly
    def load(self, conversation_id: str) -> list[Message]: ...

def sqlite_history(url: str, *, max_connections: int = 8) -> RustHistoryStore: ...
