"""Python exiting right after a run: Rust threads may still be handing results to Python."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from factories import text_response
from fake_openai import FakeOpenAI

SCRIPT = """
import asyncio
from pyox_agents import Agent, Client

agent = Agent(model="test-model", client=Client(api_key="test-key", base_url="{base_url}"))
print(asyncio.run(agent.run("Hi")).text)
"""


def test_a_script_that_ends_right_after_its_run_exits_cleanly(fake_openai: FakeOpenAI) -> None:
    """This crashed about 1 exit in 20 on Python 3.11 to 3.13, so the script runs many times."""
    runs = 24
    fake_openai.queue(*[text_response("Hi") for _ in range(runs)])
    script = SCRIPT.format(base_url=fake_openai.base_url)

    def run_script(_: int) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "PYTHONFAULTHANDLER": "1"}
        return subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30)

    with ThreadPoolExecutor(8) as pool:
        finished = list(pool.map(run_script, range(runs)))

    assert [(process.returncode, process.stdout, process.stderr) for process in finished] == [(0, "Hi\n", "")] * runs
