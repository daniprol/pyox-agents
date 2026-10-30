"""Examples, run as scripts against the fake API."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from factories import text_response, tool_call, tool_calls_response
from fake_openai import FakeOpenAI

EXAMPLES = Path(__file__).parents[1] / "examples"


# Each history example runs twice, like a restart.
@pytest.mark.parametrize("script", ["python_history.py", "rust_history.py"])
def test_history_examples_remember_across_restarts(fake_openai: FakeOpenAI, tmp_path: Path, script: str) -> None:
    fake_openai.queue(*(text_response(f"answer {number}") for number in range(4)))
    environment = {**os.environ, "OPENAI_BASE_URL": fake_openai.base_url, "OPENAI_API_KEY": "test-key"}

    for _ in range(2):
        subprocess.run([sys.executable, EXAMPLES / script], cwd=tmp_path, env=environment, check=True)

    # The first request after the "restart" carries the system prompt, the 4 saved messages and the new prompt.
    restarted_request = fake_openai.requests[2].body["messages"]
    assert [message["content"] for message in restarted_request] == [
        "Be brief.",
        "My favourite city is Vigo.",
        "answer 0",
        "What is my favourite city?",
        "answer 1",
        "My favourite city is Vigo.",
    ]


def test_images_and_files_example(fake_openai: FakeOpenAI, tmp_path: Path) -> None:
    fake_openai.queue(text_response("A cat"), text_response("A short report"))
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.7 fake document")
    environment = {**os.environ, "OPENAI_BASE_URL": fake_openai.base_url, "OPENAI_API_KEY": "test-key"}

    output = subprocess.run(
        [sys.executable, EXAMPLES / "images_and_files.py", "report.pdf"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert output == "A cat\nA short report\n"
    image_part = fake_openai.requests[0].body["messages"][1]["content"][1]
    file_part = fake_openai.requests[1].body["messages"][1]["content"][1]
    assert image_part["type"] == "image_url"
    assert file_part["file"]["filename"] == "report.pdf"


def test_human_in_the_loop_example(fake_openai: FakeOpenAI, tmp_path: Path) -> None:
    fake_openai.queue(
        tool_calls_response(tool_call("book_table", restaurant="Casa Marcelo")),
        text_response("Booked for 21:30"),
        tool_calls_response(tool_call("delete_file", path="old.log")),
        text_response("Deleted"),
    )
    environment = {**os.environ, "OPENAI_BASE_URL": fake_openai.base_url, "OPENAI_API_KEY": "test-key"}

    output = subprocess.run(
        [sys.executable, EXAMPLES / "human_in_the_loop.py"],
        cwd=tmp_path,
        env=environment,
        input="21:30\ny\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert output.endswith("Booked for 21:30\nAllow delete_file {'path': 'old.log'}? [y/n] Deleted\n")
    assert fake_openai.requests[1].body["messages"][-1]["content"] == "Booked Casa Marcelo at 21:30"
    assert fake_openai.requests[3].body["messages"][-1]["content"] == "Deleted old.log"
