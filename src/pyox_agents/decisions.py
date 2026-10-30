"""OpenAI's Decisions API: fast, cheap answers to questions about an input, as probabilities.

    decision = await decide(
        "The deploy failed twice and customers are seeing 500s. Can someone look now?",
        {
            "urgent": Predicate("Does this need attention right now?"),
            "team": Choice("Which team should handle it?", {"infra": "Deploys and outages.", "billing": "Payments."}),
            "severity": Score("How severe is the impact?", ["Cosmetic", "Degraded", "Outage"]),
        },
    )
    decision["urgent"].probability  # 0.97
    decision["team"].choice  # "infra"
    decision["severity"].score  # 1.8 (0 is "Cosmetic", 2 is "Outage")

The input is text, a prompt (text and images, as for `agent.run`) or a history
(`result.messages`), so you can ask about a whole conversation. Any answer can
be a `Refusal` instead: the model declined that question.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pyox_agents._core import Client, Decision, send_decision
from pyox_agents.messages import Message, Prompt

DEFAULT_MODEL = "gpt-6-luna"


@dataclass(frozen=True, slots=True)
class Predicate:
    """Is a statement true? The answer is a `PredicateAnswer`: its `probability`, from 0 to 1."""

    instructions: str

    def _to_json(self, name: str) -> dict[str, Any]:
        return {"type": "predicate", "name": name, "instructions": self.instructions}


@dataclass(frozen=True, slots=True)
class Choice:
    """Which option fits best? The answer is a `ChoiceAnswer`: its `choice`, `confidence` and `probabilities`.

    `choices` lists the options (strings or booleans), or maps each one to a description:
    `Choice("Which team?", {"infra": "Deploys and outages.", "billing": "Payments."})`.
    """

    instructions: str
    choices: Sequence[str | bool] | Mapping[str | bool, str]

    def _to_json(self, name: str) -> dict[str, Any]:
        return {
            "type": "choice",
            "name": name,
            "instructions": self.instructions,
            "choices": _options(self.choices, "value"),
        }


@dataclass(frozen=True, slots=True)
class Score:
    """Where on a scale? The answer is a `ScoreAnswer`: its `score`, `confidence` and `probabilities`.

    `levels` go from the lowest to the highest: labels, or labels mapped to descriptions.
    The `score` is the probability-weighted average of the level indices (0 is the lowest).
    """

    instructions: str
    levels: Sequence[str] | Mapping[str, str]

    def _to_json(self, name: str) -> dict[str, Any]:
        return {
            "type": "score",
            "name": name,
            "instructions": self.instructions,
            "levels": _options(self.levels, "label"),
        }


Question = Predicate | Choice | Score


def _options(options: Sequence[Any] | Mapping[Any, str], key: str) -> list[dict[str, Any]]:
    """`[{key: option}]`, with a `"description"` when the options map to descriptions."""
    if isinstance(options, Mapping):
        return [{key: option, "description": description} for option, description in options.items()]
    return [{key: option} for option in options]


async def decide(
    input: Prompt | Sequence[Message],  # noqa: A002 (the API's name)
    questions: Mapping[str, Question],
    *,
    client: Client | None = None,
    model: str = DEFAULT_MODEL,
    safety_identifier: str | None = None,
    prompt_cache_key: str | None = None,
) -> Decision:
    """Asks the Decisions API `questions` about `input`, all in one request.

    `questions` maps a name to each question; the answers come back under the same
    names. `client` defaults to `Client()`, like an agent's. `safety_identifier` (a
    hashed user id) and `prompt_cache_key` are passed to the API as is.
    """
    return await send_decision(
        client or Client(),
        input,
        [question._to_json(name) for name, question in questions.items()],
        model=model,
        safety_identifier=safety_identifier,
        prompt_cache_key=prompt_cache_key,
    )
