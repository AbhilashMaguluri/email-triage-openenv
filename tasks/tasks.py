"""
OpenEnv Email Triage – Tasks & Graders
=======================================

Three difficulty tiers evaluate an agent's ability to triage emails:

* **Easy**   – classify the email category            (1 criterion)
* **Medium** – classify + set priority                 (2 criteria)
* **Hard**   – classify + set priority + draft reply   (3 criteria)

Each task ships with a *deterministic* grader that inspects the
environment state after execution and returns a score in [0.0, 1.0].

Public API
----------
- ``get_task(difficulty)``   → ``Task``
- ``run_task(difficulty)``   → ``TaskResult``
- ``run_all_tasks()``        → ``dict[str, TaskResult]``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from env.environment import EmailTriageEnv
from env.models import Action


# ──────────────────────────────────────────────────────────────
# Enums & Type Aliases
# ──────────────────────────────────────────────────────────────

class TaskDifficulty(str, Enum):
    """Supported difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# A Grader receives the environment state dict and returns 0.0–1.0.
Grader = Callable[[dict[str, Any]], float]


# ──────────────────────────────────────────────────────────────
# Grader Functions (deterministic, pure)
# ──────────────────────────────────────────────────────────────

def _grade_easy(state: dict[str, Any]) -> float:
    """Score 1.0 if the category matches, else 0.0."""
    obs = state["observation"]
    expected = state["expected"]
    return 1.0 if obs.get("category") == expected["category"] else 0.0


def _grade_medium(state: dict[str, Any]) -> float:
    """
    Score breakdown (max 1.0):
        +0.5  correct category
        +0.5  correct priority
    """
    obs = state["observation"]
    expected = state["expected"]
    score = 0.0
    if obs.get("category") == expected["category"]:
        score += 0.5
    if obs.get("priority") == expected["priority"]:
        score += 0.5
    return score


def _grade_hard(state: dict[str, Any]) -> float:
    """
    Score breakdown (max 1.0):
        +0.4  correct category
        +0.3  correct priority
        +0.3  correct reply (exact match)
    """
    obs = state["observation"]
    expected = state["expected"]
    score = 0.0
    if obs.get("category") == expected["category"]:
        score += 0.4
    if obs.get("priority") == expected["priority"]:
        score += 0.3
    if obs.get("reply") == expected["reply"]:
        score += 0.3
    return score


# ──────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskResult:
    """Immutable result produced after a task is executed and graded."""
    task_name: str
    difficulty: TaskDifficulty
    score: float           # grader output in [0.0, 1.0]
    max_score: float       # always 1.0 for normalised grading
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when the agent achieved the maximum possible score."""
        return self.score >= self.max_score

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"TaskResult(name={self.task_name!r}, difficulty={self.difficulty.value}, "
            f"score={self.score:.2f}/{self.max_score:.2f}, status={status})"
        )


@dataclass(frozen=True)
class Task:
    """
    A single evaluation task.

    Parameters
    ----------
    name : str
        Human-readable task name.
    difficulty : TaskDifficulty
        easy / medium / hard.
    actions : list[Action]
        Actions the agent will take.  Use ``content="__FROM_EXPECTED__"``
        to fill in the correct answer from the environment at runtime
        (useful for gold-standard / oracle runs).
    grader : Grader
        Deterministic function ``(state) -> float``.
    max_score : float
        Upper bound for the grader output (default 1.0).
    """
    name: str
    difficulty: TaskDifficulty
    actions: list[Action]
    grader: Grader
    max_score: float = 1.0

    def execute(self, env: EmailTriageEnv) -> TaskResult:
        """
        Reset the environment, step through all actions, then grade.

        Returns a ``TaskResult`` with the score and diagnostic details.
        """
        env.reset()
        expected = env.state()["expected"]

        # Resolve placeholder actions into concrete ones
        actions = _resolve_actions(self.actions, self.difficulty, expected)

        for action in actions:
            result = env.step(action)
            if result.done:
                break

        state = env.state()
        score = self.grader(state)

        return TaskResult(
            task_name=self.name,
            difficulty=self.difficulty,
            score=round(score, 2),
            max_score=self.max_score,
            details={
                "observation": state["observation"],
                "expected": state["expected"],
                "total_reward": state["total_reward"],
                "score_breakdown": _score_breakdown(state, self.difficulty),
            },
        )


# ──────────────────────────────────────────────────────────────
# Internal Helpers
# ──────────────────────────────────────────────────────────────

_ACTION_FIELD_MAP: dict[str, str] = {
    "classify_email": "category",
    "set_priority": "priority",
    "generate_reply": "reply",
}


def _expected_content_for(action_type: str, expected: dict[str, str]) -> str:
    """Look up the gold-standard value for a given action type."""
    return expected[_ACTION_FIELD_MAP[action_type]]


def _resolve_actions(
    actions: list[Action],
    difficulty: TaskDifficulty,
    expected: dict[str, str],
) -> list[Action]:
    """
    Replace ``__FROM_EXPECTED__`` sentinel values with actual expected
    content so oracle / gold-standard runs work automatically.
    """
    resolved: list[Action] = []
    for action in actions:
        if action.content == "__FROM_EXPECTED__":
            content = _expected_content_for(action.action_type, expected)
            resolved.append(Action(action_type=action.action_type, content=content))
        else:
            resolved.append(action)
    return resolved


def _score_breakdown(
    state: dict[str, Any],
    difficulty: TaskDifficulty,
) -> dict[str, dict[str, Any]]:
    """
    Build a human-readable breakdown showing each criterion, whether it
    matched, and its weight contribution.
    """
    obs = state["observation"]
    expected = state["expected"]
    breakdown: dict[str, dict[str, Any]] = {}

    weights = {
        TaskDifficulty.EASY: {"category": 1.0},
        TaskDifficulty.MEDIUM: {"category": 0.5, "priority": 0.5},
        TaskDifficulty.HARD: {"category": 0.4, "priority": 0.3, "reply": 0.3},
    }

    for criterion, weight in weights[difficulty].items():
        actual = obs.get(criterion)
        exp = expected.get(criterion)
        matched = actual == exp
        breakdown[criterion] = {
            "weight": weight,
            "matched": matched,
            "earned": weight if matched else 0.0,
            "actual": actual,
            "expected": exp,
        }

    return breakdown


# ──────────────────────────────────────────────────────────────
# Pre-built Task Definitions
# ──────────────────────────────────────────────────────────────

TASK_EASY = Task(
    name="Email Classification",
    difficulty=TaskDifficulty.EASY,
    actions=[
        Action(action_type="classify_email", content="__FROM_EXPECTED__"),
    ],
    grader=_grade_easy,
)

TASK_MEDIUM = Task(
    name="Email Classification + Priority",
    difficulty=TaskDifficulty.MEDIUM,
    actions=[
        Action(action_type="classify_email", content="__FROM_EXPECTED__"),
        Action(action_type="set_priority", content="__FROM_EXPECTED__"),
    ],
    grader=_grade_medium,
)

TASK_HARD = Task(
    name="Full Email Triage Pipeline",
    difficulty=TaskDifficulty.HARD,
    actions=[
        Action(action_type="classify_email", content="__FROM_EXPECTED__"),
        Action(action_type="set_priority", content="__FROM_EXPECTED__"),
        Action(action_type="generate_reply", content="__FROM_EXPECTED__"),
    ],
    grader=_grade_hard,
)


# ──────────────────────────────────────────────────────────────
# Task Registry & Public API
# ──────────────────────────────────────────────────────────────

TASK_REGISTRY: dict[TaskDifficulty, Task] = {
    TaskDifficulty.EASY: TASK_EASY,
    TaskDifficulty.MEDIUM: TASK_MEDIUM,
    TaskDifficulty.HARD: TASK_HARD,
}


def get_task(difficulty: TaskDifficulty | str) -> Task:
    """Retrieve a task by its difficulty level."""
    if isinstance(difficulty, str):
        difficulty = TaskDifficulty(difficulty)
    return TASK_REGISTRY[difficulty]


def run_task(
    difficulty: TaskDifficulty | str,
    env: EmailTriageEnv | None = None,
) -> TaskResult:
    """Run a single task on the given (or a fresh) environment."""
    task = get_task(difficulty)
    env = env or EmailTriageEnv()
    return task.execute(env)


def run_all_tasks(
    env: EmailTriageEnv | None = None,
) -> dict[str, TaskResult]:
    """
    Run every registered task and return a mapping of
    ``{ difficulty_value: TaskResult }``.

    Example
    -------
    >>> scores = run_all_tasks()
    >>> for level, result in scores.items():
    ...     print(f"{level}: {result.score}")
    easy: 1.0
    medium: 1.0
    hard: 1.0
    """
    env = env or EmailTriageEnv()
    return {
        difficulty.value: task.execute(env)
        for difficulty, task in TASK_REGISTRY.items()
    }
