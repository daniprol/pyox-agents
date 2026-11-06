# Skills

A skill is a set of instructions, and sometimes tools, for one kind of task: filling PDF forms, writing sales
reports, handling refunds. The model always sees a short catalog, one line per skill. When a task fits a skill,
it loads the skill and only then reads its instructions and gets its tools. Dozens of skills cost a few tokens each
until the model uses one.

```python
from pyox_agents import Agent
from pyox_agents.skills import Skill, skills_from_dir

refunds = Skill(
    name="refunds",
    description="Refund orders. Use when a customer asks for their money back.",
    instructions="Check the order with get_order first. Refund at most 100 €; above that, ask a person.",
    tools=[get_order, refund],  # hidden until the model loads the skill
)

agent = Agent(
    model="gpt-5-mini",
    instructions="You are the shop's support agent.",
    skills=[*skills_from_dir(".agents/skills"), refunds],
)
```

## How the model uses them

`Agent(skills=...)` adds two tools and a hook:

1. **The catalog.** Before each request, the hook adds a `## Skills` section to the instructions, with each
   skill's name and description. The model picks skills by their description alone, so say what the skill does
   *and when to use it*.
2. **`load_skill(name)`.** The model loads a skill when a task fits it. The tool gives back the skill's
   instructions, the names of its files, and the tools it brings.
3. **`read_skill_file(skill, path)`.** The model reads one of a skill's files, such as `references/forms.md`. This
   tool is only there when some skill has files.
4. **Hidden tools.** A skill's tools are left out of every request until the model loads the skill. A call to a
   hidden tool fails with `Unknown tool "..."`, as for a tool that doesn't exist.

A skill counts as loaded once a `load_skill` call for it succeeded. pyox reads that from the conversation, so a
skill stays loaded in later runs of the same `history` or `conversation_id`. If you trim old messages from the
history, the skills loaded in them are unloaded too.

## Skill folders (`SKILL.md`)

`skills_from_dir(folder)` reads skills in the [Agent Skills](https://agentskills.io) format, which most coding
agents share. Each subfolder with a `SKILL.md` file is one skill:

```text
.agents/skills/
└── pdf-forms/
    ├── SKILL.md
    ├── references/fields.md
    └── scripts/fill.py
```

```markdown
---
name: pdf-forms
description: Fill PDF forms. Use when the user sends a PDF form or asks to fill one.
---
# Filling PDF forms

1. Read `references/fields.md` for the names of the fields.
2. ...
```

The frontmatter needs a `name` (one word, the folder's name) and a `description`. Other keys (`license`,
`metadata`, ...) are kept in `skill.metadata`. The other files are the skill's `files`, read as text when the model
asks. pyox never runs a skill's scripts: give the skill tools to do that work.

```python
skills = skills_from_dir("skills", tools={"pdf-forms": [fill_pdf_form]})
```

`tools=` gives skills from a folder their tools, by skill name. A folder can also be inside your package
(`importlib.resources.files("my_app") / "skills"`) or a zip file (`zipfile.Path("skills.zip")`). A `SKILL.md`
that cannot be read is skipped with a warning, so one broken skill doesn't stop your app.

## Skills in code

`Skill(...)` makes a skill in code. `instructions`, and each of `files`, is text or a function (sync or async) that
pyox calls when the model asks for it:

```python
Skill(
    name="sql-reports",
    description="Write SQL reports on the sales database. Use for questions about sales.",
    instructions=lambda: Path("prompts/sql.md").read_text(),
    files={"schema.sql": load_schema},  # an async function, say
    tools=[run_query],
)
```

`Skill.from_skill_md(text, tools=..., files=...)` makes one from the text of a `SKILL.md` file you keep
somewhere else.

## Skills from anywhere (`SkillLike`)

`Agent(skills=...)` takes any object with these attributes, so skills can live in a database, an object store or
behind an API. The catalog is read once, when you make the agent. The rest is read only when the model asks:

```python
class SkillLike(Protocol):
    name: str
    description: str
    tools: Sequence[ToolLike | Callable]  # plain functions are wrapped with @tool
    files: Collection[str]  # the paths read_file accepts

    def load_instructions(self) -> str | Awaitable[str]: ...
    def read_file(self, path: str) -> str | Awaitable[str]: ...
```

For example, skills kept in a table:

```python
@dataclass
class DatabaseSkill:
    name: str
    description: str
    pool: asyncpg.Pool
    files: tuple[str, ...] = ()
    tools: tuple[Tool, ...] = ()

    async def load_instructions(self) -> str:
        return await self.pool.fetchval("SELECT body FROM skills WHERE name = $1", self.name)

    async def read_file(self, path: str) -> str:
        text = await self.pool.fetchval("SELECT body FROM skill_files WHERE skill = $1 AND path = $2", self.name, path)
        if text is None:
            raise ToolFailed(f"No file {path!r}")  # the model reads this message
        return text


rows = await pool.fetch("SELECT name, description FROM skills")
agent = Agent(model="gpt-5-mini", skills=[DatabaseSkill(row["name"], row["description"], pool) for row in rows])
```

Raise `ToolFailed` in `load_instructions` or `read_file` to tell the model what went wrong. A failed load loads
nothing: the skill's tools stay hidden.

## Notes

- Skill names and the names of their tools must be unique, and differ from the agent's own tools.
- Skills are the agent's, not the run's: make the agent with the skills a user may use.
- [Subagents](subagents.md) don't get the calling agent's skills: give them their own `skills=`. With
  `inherit_tools=True` they get `load_skill` and every skill's tools, none of them hidden.
- The hook runs before your own [`before_model_request`](hooks.md) hooks, which see the catalog in
  `request["instructions"]` and only the tools offered.
