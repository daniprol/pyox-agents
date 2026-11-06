"""Skills: instructions (and tools) the model loads only when a task needs them.

The model always sees a short catalog, one line per skill (its name and
description). When a task fits a skill, it calls `load_skill` and reads the
skill's instructions; files the skill lists come one at a time with
`read_skill_file`. A skill's own tools are hidden until the model loads it, so
dozens of skills cost a few tokens each until they are used.

    from pyox_agents import Agent
    from pyox_agents.skills import Skill, skills_from_dir

    agent = Agent(
        model="gpt-5-mini",
        skills=[
            *skills_from_dir("skills"),  # folders with a SKILL.md (agentskills.io)
            Skill(name="refunds", description="Refund an order.", instructions="...", tools=[refund]),
        ],
    )

Skills can come from anywhere: `Skill` takes text or functions (sync or async)
that load it, and any object with the attributes of `SkillLike` works too (a
skill read from a database, say).
"""

import inspect
import json
import warnings
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources.abc import Traversable
from os import PathLike
from pathlib import Path
from typing import Any, Protocol

from pyox_agents._core import RunContext, parse_skill_md
from pyox_agents.hooks import ModelRequest
from pyox_agents.messages import Message
from pyox_agents.tools import Tool, ToolFailed, ToolLike, tool

LOAD_SKILL = "load_skill"
READ_SKILL_FILE = "read_skill_file"

Text = str | Callable[[], str | Awaitable[str]]
"""Text, or a function (sync or async) that gives it when the model asks for it."""


class SkillLike(Protocol):
    """What the agent needs from a skill. `Skill` fits; so does any object with
    these attributes, to load skills from a database, an API or anywhere else.

    `name` and `description` make the catalog the model always sees. The rest is
    read only when the model asks: `load_instructions()` when it loads the skill,
    `read_file(path)` for one of the `files`. Both may be sync or async, and raise
    `ToolFailed` to tell the model what went wrong. `tools` are hidden until the
    model loads the skill.
    """

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def tools(self) -> Sequence[ToolLike | Callable[..., Any]]: ...  # plain functions are wrapped with `@tool`
    @property
    def files(self) -> Collection[str]: ...
    def load_instructions(self) -> str | Awaitable[str]: ...
    def read_file(self, path: str) -> str | Awaitable[str]: ...


@dataclass(frozen=True, slots=True)
class Skill:
    """A skill made in code. `instructions` and each of `files` are text, or a
    function (sync or async) called when the model asks for them:

        Skill(
            name="sql-reports",
            description="Write SQL reports on the sales database. Use for sales questions.",
            instructions=lambda: (Path(__file__).parent / "sql.md").read_text(),
            files={"schema.sql": load_schema},
            tools=[run_query],  # plain functions are wrapped with `@tool`
        )

    The model sees `files` by name, and reads them with `read_skill_file`.
    `metadata` is for your app (a `SKILL.md`'s frontmatter, for example); the
    model never sees it.
    """

    name: str
    description: str
    instructions: Text
    tools: Sequence[ToolLike | Callable[..., Any]] = ()
    files: Mapping[str, Text] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or any(character.isspace() for character in self.name):
            raise ValueError(f"A skill's name must be one word, like 'pdf-forms': got {self.name!r}")
        if not self.description.strip():
            raise ValueError(f"Skill {self.name!r} has no description: the model picks skills by it")

    def load_instructions(self) -> str | Awaitable[str]:
        return _text(self.instructions)

    def read_file(self, path: str) -> str | Awaitable[str]:
        if path not in self.files:
            raise ToolFailed(f"Skill {self.name!r} has no file {path!r}. Its files: {', '.join(self.files) or 'none'}")
        return _text(self.files[path])

    @classmethod
    def from_skill_md(
        cls,
        text: str,
        *,
        tools: Sequence[ToolLike | Callable[..., Any]] = (),
        files: Mapping[str, Text] | None = None,
    ) -> "Skill":
        """A skill from the text of a `SKILL.md` file (from a database, say): the
        frontmatter gives `name`, `description` and `metadata`, and the Markdown
        after it is the instructions. Raises `ValueError` if the frontmatter is invalid."""
        frontmatter, body = parse_skill_md(text)
        name, description = frontmatter.get("name"), frontmatter.get("description")
        if not isinstance(name, str) or not isinstance(description, str):
            raise ValueError("The frontmatter of SKILL.md needs a `name` and a `description` (text)")
        return cls(name, description.strip(), body, tools=tools, files=files or {}, metadata=frontmatter)


def skills_from_dir(
    root: str | PathLike[str] | Traversable,
    *,
    tools: Mapping[str, Sequence[ToolLike | Callable[..., Any]]] | None = None,
) -> list[Skill]:
    """The skills in a folder: each subfolder with a `SKILL.md` file is one
    (the [Agent Skills](https://agentskills.io) format, as in `.agents/skills`).

        skills_from_dir("skills", tools={"pdf-forms": [fill_pdf_form]})

    The other files in a skill's folder (`references/forms.md`, `scripts/fill.py`...)
    are its `files`: the model reads them with `read_skill_file`, but never runs them.
    `tools` gives skills tools, by skill name. `root` may also be any `Traversable`:
    files inside your package (`importlib.resources.files("my_app") / "skills"`) or a
    zip file (`zipfile.Path`). A `SKILL.md` that cannot be read is skipped with a warning.
    """
    folder = Path(root) if isinstance(root, (str, PathLike)) else root
    tools = dict(tools or {})
    skills = []
    for skill_folder in sorted(folder.iterdir(), key=lambda item: item.name):
        skill_md = skill_folder / "SKILL.md"
        if skill_folder.name.startswith(".") or not skill_folder.is_dir() or not skill_md.is_file():
            continue
        try:
            skill = Skill.from_skill_md(
                skill_md.read_text(encoding="utf-8"),
                tools=tools.pop(skill_folder.name, ()),
                files={path: _file_reader(file) for path, file in _files(skill_folder)},
            )
        except ValueError as error:
            warnings.warn(f"Skipping the skill in {skill_folder}: {error}", stacklevel=2)
            continue
        if skill.name != skill_folder.name:
            warnings.warn(f"The skill in {skill_folder} is named {skill.name!r}, not after its folder", stacklevel=2)
        skills.append(skill)
    if tools:
        raise ValueError(f"No skill folder in {folder} for the tools of: {', '.join(tools)}")
    return skills


class SkillsHook:
    """What `Agent(skills=...)` adds: the `load_skill` and `read_skill_file` tools,
    the skills' own tools, and a `before_model_request` hook. The hook adds the
    catalog to the instructions and hides the tools of skills not loaded yet.

    A skill counts as loaded once a `load_skill` call for it succeeded in the
    conversation, so it stays loaded across runs of the same history.
    """

    def __init__(self, skills: Sequence[SkillLike]) -> None:
        self._skills: dict[str, SkillLike] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise ValueError(f"Two skills are named {skill.name!r}")
            self._skills[skill.name] = skill
        names = list(self._skills)
        with_files = [skill.name for skill in skills if skill.files]
        self.tools: list[ToolLike] = [_load_skill_tool(self._load_skill, names)]
        if with_files:
            self.tools.append(_read_skill_file_tool(self._read_skill_file, with_files))
        # The names of each skill's tools, hidden until the model loads the skill.
        self._skill_tools: dict[str, list[str]] = {}
        for skill in skills:
            skill_tools = [_as_tool(item) for item in skill.tools]
            self.tools.extend(skill_tools)
            self._skill_tools[skill.name] = [skill_tool.name for skill_tool in skill_tools]
        catalog = "\n".join(f"- {skill.name}: {skill.description}" for skill in skills)
        self.catalog = (
            "## Skills\n\nSkills hold instructions, and sometimes tools, for specific tasks. "
            f"When a task fits a skill's description, call {LOAD_SKILL} with its name first "
            f"and follow what it says.\n\n{catalog}"
        )

    def before_model_request(self, ctx: RunContext, request: ModelRequest) -> ModelRequest:
        loaded = loaded_skills(request["messages"])
        hidden = {name for skill, tool_names in self._skill_tools.items() if skill not in loaded for name in tool_names}
        instructions = f"{request['instructions']}\n\n{self.catalog}" if request["instructions"] else self.catalog
        tools = [name for name in request["tools"] if name not in hidden]
        return {**request, "instructions": instructions, "tools": tools}

    async def _load_skill(self, name: str) -> str:
        skill = self._skill(name)
        instructions = await _awaited(skill.load_instructions())
        parts = [f'<skill name="{skill.name}">\n{instructions.strip()}\n</skill>']
        if skill.files:
            parts.append(f"Files of this skill (read them with {READ_SKILL_FILE}): {', '.join(skill.files)}")
        if self._skill_tools[skill.name]:
            parts.append(f"Tools you can use now: {', '.join(self._skill_tools[skill.name])}")
        return "\n\n".join(parts)

    async def _read_skill_file(self, skill: str, path: str) -> str:
        return await _awaited(self._skill(skill).read_file(path.removeprefix("./")))

    def _skill(self, name: str) -> SkillLike:
        if name not in self._skills:
            raise ToolFailed(f"No skill named {name!r}. Skills: {', '.join(self._skills)}")
        return self._skills[name]


def loaded_skills(messages: Sequence[Message]) -> set[str]:
    """The names of the skills that a `load_skill` call loaded in these messages."""
    requested = {}  # call id -> skill name
    loaded = set()
    for message in messages:
        if message["role"] == "assistant":
            for call in message.get("tool_calls", []):
                arguments = call["arguments"]
                if isinstance(arguments, str):  # arguments that were not valid JSON stay text
                    arguments = _json_or_none(arguments)
                if call["name"] == LOAD_SKILL and isinstance(arguments, dict):
                    requested[call["id"]] = arguments.get("name")
        elif message["role"] == "tool" and message["tool_call_id"] in requested and not message.get("is_error"):
            loaded.add(requested[message["tool_call_id"]])
    return loaded


def _load_skill_tool(function: Callable[..., Awaitable[str]], names: list[str]) -> Tool:
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": names, "description": "The skill's name"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    description = "Load a skill: its instructions, and any tools it brings. Skills are listed in your instructions."
    return Tool(LOAD_SKILL, description, parameters, function)


def _read_skill_file_tool(function: Callable[..., Awaitable[str]], names: list[str]) -> Tool:
    parameters = {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "enum": names, "description": "The skill's name"},
            "path": {"type": "string", "description": "The file's path, as the skill lists it"},
        },
        "required": ["skill", "path"],
        "additionalProperties": False,
    }
    return Tool(READ_SKILL_FILE, "Read one of the files a loaded skill lists.", parameters, function)


def _files(folder: Traversable, prefix: str = "") -> Iterator[tuple[str, Traversable]]:
    """Every file under `folder` (but the top `SKILL.md` and hidden ones), by relative path."""
    for item in sorted(folder.iterdir(), key=lambda item: item.name):
        path = f"{prefix}{item.name}"
        if item.name.startswith(".") or path == "SKILL.md":
            continue
        if item.is_dir():
            yield from _files(item, f"{path}/")
        else:
            yield path, item


def _file_reader(file: Traversable) -> Callable[[], str]:
    def read() -> str:
        try:
            return file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolFailed(f"{file.name} is not a text file") from None

    return read


def _json_or_none(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _text(text: Text) -> str | Awaitable[str]:
    return text if isinstance(text, str) else text()


async def _awaited(value: str | Awaitable[str]) -> str:
    return await value if inspect.isawaitable(value) else value


def _as_tool(item: ToolLike | Callable[..., Any]) -> ToolLike:
    return item if hasattr(item, "invoke") else tool(item)  # ty: ignore[invalid-return-type]
