"""The history examples, run as scripts against the fake API: each one twice, like a restart."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from factories import text_response
from fake_openai import FakeOpenAI

EXAMPLES = Path(__file__).parents[1] / "examples"


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
