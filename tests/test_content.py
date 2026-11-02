"""Images and files in prompts: how they are built, sent to the API and kept in the history."""

import base64
from pathlib import Path

import pytest

from conftest import MakeAgent
from factories import (
    collect_events,
    events_of_type,
    finished_calls,
    responses_answer,
    run_finished,
    text_response,
    tool_call,
    tool_calls_response,
)
from fake_openai import FakeOpenAI
from pyox_agents import (
    Hooks,
    ImagePart,
    RunContext,
    ToolCall,
    ToolFinished,
    ToolResult,
    ToolReturn,
    file,
    image,
    tool,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n fake image"
PDF_BYTES = b"%PDF-1.7 fake document"


def data_url(media_type: str, data: bytes) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


@pytest.fixture
def report_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "report.pdf"
    path.write_bytes(PDF_BYTES)
    return path


def test_image_from_a_path_bytes_or_web_address(tmp_path: Path) -> None:
    path = tmp_path / "cat.png"
    path.write_bytes(PNG_BYTES)

    assert image(path) == {"type": "image", "url": data_url("image/png", PNG_BYTES)}
    assert image(str(path)) == image(path)
    assert image(PNG_BYTES, media_type="image/png") == image(path)
    assert image("https://example.com/cat.png") == {"type": "image", "url": "https://example.com/cat.png"}


def test_file_from_a_path_bytes_or_web_address(report_pdf: Path) -> None:
    expected_url = data_url("application/pdf", PDF_BYTES)

    assert file(report_pdf) == {"type": "file", "url": expected_url, "filename": "report.pdf"}
    assert file(PDF_BYTES, media_type="application/pdf") == {
        "type": "file",
        "url": expected_url,
        "filename": "file.pdf",
    }
    assert file("https://example.com/docs/a.pdf?v=2")["filename"] == "a.pdf"
    assert file(PDF_BYTES, media_type="application/pdf", filename="q3.pdf")["filename"] == "q3.pdf"


def test_bytes_need_a_media_type() -> None:
    with pytest.raises(ValueError, match="media_type"):
        image(PNG_BYTES)


async def test_prompt_parts_are_sent_as_chat_content_parts(
    fake_openai: FakeOpenAI, make_agent: MakeAgent, report_pdf: Path
) -> None:
    fake_openai.queue(text_response("A cat and a report"))

    await make_agent().run(["What are these?", image("https://example.com/cat.png"), file(report_pdf)])

    assert fake_openai.requests[0].body["messages"][0] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            {
                "type": "file",
                "file": {"file_data": data_url("application/pdf", PDF_BYTES), "filename": "report.pdf"},
            },
        ],
    }


async def test_parts_are_kept_in_the_history(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(text_response("A cat"), text_response("Grey"))
    agent = make_agent()
    first = await agent.run(["What is this?", image("https://example.com/cat.png")])

    assert first.messages[0] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image", "url": "https://example.com/cat.png"},
        ],
    }
    await agent.run("What colour is it?", history=first.messages_json())
    assert fake_openai.requests[1].body["messages"][0]["content"][1]["type"] == "image_url"


async def test_invalid_prompt_parts_are_rejected(make_agent: MakeAgent) -> None:
    with pytest.raises(ValueError, match="Invalid content part"):
        make_agent().stream(["Hi", {"type": "video", "url": "x"}])  # ty: ignore[invalid-argument-type]


# Tool results with images and files ---------------------------------------------------------------


@tool
def screenshot(page: str) -> ToolReturn:
    """Take a screenshot of a page."""
    return ToolReturn([f"The {page} page:", image(PNG_BYTES, media_type="image/png")], metadata={"page": page})


SCREENSHOT_PARTS = [
    {"type": "text", "text": "The home page:"},
    {"type": "image", "url": data_url("image/png", PNG_BYTES)},
]


async def test_tool_results_can_hold_images(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(tool_calls_response(tool_call("screenshot", page="home")), text_response("A cat"))

    events = await collect_events(make_agent(tools=[screenshot]).stream("What is on the home page?"))

    [finished] = events_of_type(events, ToolFinished)
    assert finished.result == "The home page:"  # the text, for streams and logs
    assert finished.content == SCREENSHOT_PARTS
    assert run_finished(events).messages[2]["content"] == SCREENSHOT_PARTS
    # Chat Completions tool messages only take text: the image follows in a user message.
    sent = fake_openai.requests[1].body["messages"]
    assert sent[2] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "The home page:\n(Its images and files follow in the next message.)",
    }
    assert sent[3] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Images and files from the screenshot tool (call call_0):"},
            {"type": "image_url", "image_url": {"url": data_url("image/png", PNG_BYTES)}},
        ],
    }


async def test_the_responses_api_gets_images_in_the_tool_output(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    fake_openai.queue(responses_answer(tool_calls=(tool_call("screenshot", page="home"),)), responses_answer("A cat"))

    await make_agent(api="responses", tools=[screenshot]).run("What is on the home page?")

    [output] = [item for item in fake_openai.requests[1].body["input"] if item.get("type") == "function_call_output"]
    assert output["output"] == [
        {"type": "input_text", "text": "The home page:"},
        {"type": "input_image", "image_url": data_url("image/png", PNG_BYTES)},
    ]


async def test_a_tool_can_return_an_image_or_file_alone(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    @tool
    def chart() -> ImagePart:
        """Draw the sales chart."""
        return image(PNG_BYTES, media_type="image/png")

    @tool
    def numbers() -> list[int]:
        """Some numbers (plain data is still sent as JSON)."""
        return [1, 2]

    fake_openai.queue(tool_calls_response(tool_call("chart"), tool_call("numbers")), text_response("Done"))

    events = await collect_events(make_agent(tools=[chart, numbers]).stream("Chart"))

    contents = [event.content for event in finished_calls(events)]
    assert contents == [[{"type": "image", "url": data_url("image/png", PNG_BYTES)}], "[1, 2]"]


async def test_hooks_see_and_change_the_parts(fake_openai: FakeOpenAI, make_agent: MakeAgent) -> None:
    def text_only(ctx: RunContext, call: ToolCall, result: ToolResult) -> ToolResult:
        assert result["content"] == SCREENSHOT_PARTS
        return {**result, "content": "(screenshots are disabled)"}

    fake_openai.queue(tool_calls_response(tool_call("screenshot", page="home")), text_response("Ok"))

    result = await make_agent(tools=[screenshot], hooks=[Hooks(after_tool_call=text_only)]).run("Look")

    assert result.messages[2]["content"] == "(screenshots are disabled)"
