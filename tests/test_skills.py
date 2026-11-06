"""Skills (`Agent(skills=...)`, `pyox_agents.skills`): a catalog in the instructions,
`load_skill` and `read_skill_file`, and skill tools hidden until the skill is loaded."""

import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from conftest import MakeAgent
from factories import collect_events, finished_calls, run_finished, text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI, RecordedRequest
from pyox_agents import ToolFailed, ToolLike, tool
from pyox_agents.skills import Skill, SkillLike, loaded_skills, skills_from_dir


def refund(order_id: str) -> str:
    """Refund an order."""
    return f"Refunded {order_id}"


def refunds_skill(**options: Any) -> Skill:
    options = {"instructions": "Check the order first, then call refund.", "tools": [refund], **options}
    return Skill(name="refunds", description="Refund orders. Use when a customer asks for their money back.", **options)


def offered_tools(request: RecordedRequest) -> list[str]:
    return [offered["function"]["name"] for offered in request.body.get("tools", [])]


def system_prompt(request: RecordedRequest) -> str:
    return request.body["messages"][0]["content"]


def write_skill(root: Path, folder: str, skill_md: str, files: dict[str, str | bytes] | None = None) -> None:
    (root / folder).mkdir(parents=True)
    (root / folder / "SKILL.md").write_text(skill_md)
    for path, content in (files or {}).items():
        file = root / folder / path
        file.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            file.write_bytes(content)
        else:
            file.write_text(content)


async def test_a_skills_tools_appear_once_the_model_loads_it(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("load_skill", name="refunds")),
        tool_calls_response(tool_call("refund", order_id="A1")),
        text_response("Refunded."),
    )
    agent = make_agent(instructions="Be kind.", skills=[refunds_skill()])

    events = await collect_events(agent.stream("Money back for A1"))

    first, second, third = fake_openai.requests
    assert offered_tools(first) == ["load_skill"]
    assert system_prompt(first).startswith("Be kind.\n\n## Skills\n")
    assert "- refunds: Refund orders. Use when a customer asks for their money back." in system_prompt(first)
    assert first.body["tools"][0]["function"]["parameters"]["properties"]["name"]["enum"] == ["refunds"]
    assert offered_tools(second) == offered_tools(third) == ["load_skill", "refund"]
    loaded, refunded = finished_calls(events)
    assert loaded.result == (
        '<skill name="refunds">\nCheck the order first, then call refund.\n</skill>\n\nTools you can use now: refund'
    )
    assert (refunded.result, refunded.is_error) == ("Refunded A1", False)
    assert agent.tools == ()  # the skill tools are not the agent's own (subagents do not inherit them)


async def test_a_hidden_tool_cannot_be_called(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("refund", order_id="A1")), text_response("Sorry."))

    events = await collect_events(make_agent(skills=[refunds_skill()]).stream("Money back for A1"))

    [call] = finished_calls(events)
    assert (call.result, call.is_error) == ('Unknown tool "refund"', True)


async def test_a_skill_stays_loaded_in_later_runs_of_the_conversation(
    fake_openai: FakeOpenAI, make_agent: MakeAgent
) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("load_skill", name="refunds")), text_response("Ready."), text_response("Ok.")
    )
    agent = make_agent(skills=[refunds_skill()])

    first = run_finished(await collect_events(agent.stream("Refunds?")))
    await agent.run("And now?", history=first.messages)

    assert offered_tools(fake_openai.requests[-1]) == ["load_skill", "refund"]


def test_skills_work_in_sync_runs(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("load_skill", name="refunds")),
        tool_calls_response(tool_call("refund", order_id="A1")),
        text_response("Refunded."),
    )

    result = make_agent(skills=[refunds_skill()]).run_sync("Money back for A1")

    assert result.text == "Refunded."
    assert result.messages[-2]["content"] == "Refunded A1"


async def test_a_failed_load_loads_nothing(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    def unavailable() -> str:
        raise ToolFailed("The skills database is down")

    fake_openai.queue(tool_calls_response(tool_call("load_skill", name="refunds")), text_response("Sorry."))

    events = await collect_events(make_agent(skills=[refunds_skill(instructions=unavailable)]).stream("Refund"))

    [call] = finished_calls(events)
    assert (call.result, call.is_error) == ("The skills database is down", True)
    assert offered_tools(fake_openai.requests[-1]) == ["load_skill"]


@dataclass
class DatabaseSkill:
    """A skill kept somewhere else (a database, an API): any object with `SkillLike`'s attributes."""

    name: str
    description: str
    rows: dict[str, str]  # stands in for the database: path -> text
    tools: Sequence[ToolLike] = ()
    reads: list[str] = field(default_factory=list)

    @property
    def files(self) -> list[str]:
        return [path for path in self.rows if path != "SKILL.md"]

    async def load_instructions(self) -> str:
        self.reads.append("SKILL.md")
        return self.rows["SKILL.md"]

    async def read_file(self, path: str) -> str:
        self.reads.append(path)
        if path not in self.rows:
            raise ToolFailed(f"No {path}")
        return self.rows[path]


async def test_skills_can_come_from_anywhere(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    skill = DatabaseSkill("sales", "Sales reports.", {"SKILL.md": "Read schema.sql first.", "schema.sql": "CREATE"})
    fake_openai.queue(
        tool_calls_response(tool_call("load_skill", name="sales")),
        tool_calls_response(tool_call("read_skill_file", skill="sales", path="./schema.sql")),
        text_response("Done."),
    )
    skill_like: SkillLike = skill

    events = await collect_events(make_agent(skills=[skill_like]).stream("Sales?"))

    assert skill.reads == ["SKILL.md", "schema.sql"]  # read only when the model asked
    loaded, read = finished_calls(events)
    assert loaded.result.endswith("Files of this skill (read them with read_skill_file): schema.sql")
    assert read.result == "CREATE"


async def test_an_unknown_file_or_skill_tells_the_model(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(
        tool_calls_response(
            tool_call("read_skill_file", skill="refunds", path="nope.md"),
            tool_call("read_skill_file", skill="nope", path="a.md"),
        ),
        text_response("Sorry."),
    )
    skill = refunds_skill(files={"policy.md": "30 days."})

    events = await collect_events(make_agent(skills=[skill]).stream("Refund"))

    assert [call.result for call in finished_calls(events)] == [
        "Skill 'refunds' has no file 'nope.md'. Its files: policy.md",
        "No skill named 'nope'. Skills: refunds",
    ]


def test_skill_names_and_tool_names_must_be_unique(make_agent: MakeAgent) -> None:
    with pytest.raises(ValueError, match="Two skills are named 'refunds'"):
        make_agent(skills=[refunds_skill(), refunds_skill()])
    with pytest.raises(ValueError, match="refund"):
        make_agent(tools=[refund], skills=[refunds_skill()])


def test_a_skill_needs_a_name_and_a_description() -> None:
    with pytest.raises(ValueError, match="one word"):
        Skill(name="pdf forms", description="Fill PDF forms.", instructions="...")
    with pytest.raises(ValueError, match="no description"):
        Skill(name="pdf", description=" ", instructions="...")


def test_a_skill_from_the_text_of_a_skill_md() -> None:
    text = "---\nname: pdf\ndescription: Use when: the user sends a PDF\nlicense: MIT\n---\n# PDF\n"

    skill = Skill.from_skill_md(text, tools=[refund])

    assert (skill.name, skill.description, skill.instructions) == ("pdf", "Use when: the user sends a PDF", "# PDF\n")
    assert skill.metadata["license"] == "MIT"
    assert skill.tools == [refund]
    with pytest.raises(ValueError, match="needs a `name` and a `description`"):
        Skill.from_skill_md("---\nname: pdf\n---\n")
    with pytest.raises(ValueError, match="must start with a frontmatter block"):
        Skill.from_skill_md("# No frontmatter")


async def test_skills_from_a_folder(tmp_path: Path, fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    write_skill(
        tmp_path,
        "pdf-forms",
        "---\nname: pdf-forms\ndescription: Fill PDF forms.\n---\nRead references/fields.md.\n",
        {"references/fields.md": "Name, date.", "scripts/fill.py": "print()", ".DS_Store": "x", "logo.png": b"\x89PNG"},
    )
    write_skill(tmp_path, "notes", "---\nname: notes\ndescription: Take notes.\n---\nBe short.\n")
    (tmp_path / "not-a-skill").mkdir()
    fill_form = tool(name="fill_form")(refund)

    notes, pdf = skills_from_dir(tmp_path, tools={"pdf-forms": [fill_form]})

    assert (pdf.name, pdf.description, list(pdf.files)) == (
        "pdf-forms",
        "Fill PDF forms.",
        ["logo.png", "references/fields.md", "scripts/fill.py"],
    )
    assert (notes.name, notes.files, notes.tools) == ("notes", {}, ())
    assert pdf.tools == [fill_form]
    fake_openai.queue(
        tool_calls_response(
            tool_call("read_skill_file", skill="pdf-forms", path="references/fields.md"),
            tool_call("read_skill_file", skill="pdf-forms", path="logo.png"),
        ),
        text_response("Done."),
    )
    events = await collect_events(make_agent(skills=[pdf, notes]).stream("Fill it"))
    assert [(call.result, call.is_error) for call in finished_calls(events)] == [
        ("Name, date.", False),
        ("logo.png is not a text file", True),
    ]
    enum = fake_openai.requests[0].body["tools"][1]["function"]["parameters"]["properties"]["skill"]["enum"]
    assert enum == ["pdf-forms"]  # only skills with files


def test_skills_from_a_folder_warn_about_broken_skills(tmp_path: Path) -> None:
    write_skill(tmp_path, "broken", "---\nname: [unclosed\n---\n")
    write_skill(tmp_path, "renamed", "---\nname: other\ndescription: Something.\n---\n")

    with pytest.warns(UserWarning, match="Skipping|named") as warned:
        [skill] = skills_from_dir(tmp_path)

    assert skill.name == "other"
    assert ["Invalid YAML" in str(warning.message) for warning in warned] == [True, False]
    assert "named 'other', not after its folder" in str(warned[1].message)
    with pytest.raises(ValueError, match=r"No skill folder .* for the tools of: missing"):
        skills_from_dir(tmp_path / "renamed", tools={"missing": [refund]})


def test_skills_from_a_zip_file(tmp_path: Path) -> None:
    archive = tmp_path / "skills.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("skills/pdf/SKILL.md", "---\nname: pdf\ndescription: Read PDFs.\n---\nSteps.\n")
        zipped.writestr("skills/pdf/forms.md", "Forms.")

    [skill] = skills_from_dir(zipfile.Path(archive, "skills/"))

    assert (skill.instructions, skill.read_file("forms.md")) == ("Steps.\n", "Forms.")


def test_loaded_skills_come_from_successful_load_skill_calls() -> None:
    messages: list[Any] = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "name": "load_skill", "arguments": {"name": "pdf"}},
                {"id": "b", "name": "load_skill", "arguments": '{"name": "notes"}'},
                {"id": "c", "name": "load_skill", "arguments": {"name": "broken"}},
                {"id": "d", "name": "load_skill", "arguments": "not json"},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "name": "load_skill", "content": "..."},
        {"role": "tool", "tool_call_id": "b", "name": "load_skill", "content": "..."},
        {"role": "tool", "tool_call_id": "c", "name": "load_skill", "content": "Down", "is_error": True},
        {"role": "tool", "tool_call_id": "d", "name": "load_skill", "content": "Bad", "is_error": True},
    ]

    assert loaded_skills(messages) == {"pdf", "notes"}
