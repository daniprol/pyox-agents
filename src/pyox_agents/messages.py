"""Types for conversation history: plain dicts, typed with `TypedDict`.

`RunFinished.messages` returns these, and `history=` accepts them, so a
conversation can be stored as JSON and passed back as is.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal, NotRequired, Protocol, TypedDict


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image"]
    url: str  # a web address, or a `data:` URL with the image bytes


class FilePart(TypedDict):
    type: Literal["file"]
    url: str  # normally a `data:` URL with the file bytes
    filename: str


ContentPart = TextPart | ImagePart | FilePart
"""One part of a user message. Make images and files with `image(...)` and `file(...)`."""

Prompt = str | Sequence[str | ContentPart]
"""Text, or a list of text, images and files: `["What is this?", image("cat.png")]`."""


class UserMessage(TypedDict):
    role: Literal["user"]
    content: str | list[ContentPart]


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: Any  # normally a dict


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]


class ToolMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    name: str
    content: str
    is_error: NotRequired[bool]


Message = UserMessage | AssistantMessage | ToolMessage

History = Sequence[Message] | str | bytes
"""Message dicts, or JSON text of a list of them (parsed in Rust, without building dicts)."""


class HistoryStore(Protocol):
    """Loads (and optionally saves) conversations: `Agent(history_store=...)`.

    `load` and `save` may be sync or async. `save` is optional: add a
    `save(conversation_id: str, new_messages_json: str)` method to have the
    agent store each run's new messages, as JSON text of a list.
    """

    def load(self, conversation_id: str) -> History | Awaitable[History]: ...


LoadHistory = Callable[[str], History | Awaitable[History]]
"""A function that loads a conversation, for stores that only load."""
