"""Images and files in prompts: how they are built, sent to the API and kept in the history."""

import base64
from pathlib import Path

import pytest

from conftest import MakeAgent
from factories import text_response
from fake_openai import FakeOpenAI
from pyox_agents import file, image

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
    with pytest.raises(ValueError, match="Invalid prompt part"):
        make_agent().execute(["Hi", {"type": "video", "url": "x"}])  # ty: ignore[invalid-argument-type]
