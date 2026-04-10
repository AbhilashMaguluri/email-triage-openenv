"""
Static task and grader definitions for OpenEnv email-triage-env.

Each task is a plain callable that returns a dict.
Each grader is a plain callable that accepts any arguments and returns
a float strictly in (0, 1).
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------

_MIN_SCORE = 0.05
_MAX_SCORE = 0.95


def _clamp(score: float) -> float:
    """Ensure score is strictly within (0, 1)."""
    if score <= 0.0:
        return _MIN_SCORE
    if score >= 1.0:
        return _MAX_SCORE
    return score


def _compare(output: str, expected: str) -> float:
    out = output.strip().lower()
    exp = expected.strip().lower()
    if out == exp:
        return 0.85
    if out in exp or exp in out:
        return 0.65
    return 0.15


def _extract_output(args: tuple, kwargs: dict) -> str:
    """Best-effort extraction of the model output from any calling convention."""
    # kwargs style: grader(output="...", expected="...")
    for key in ("output", "prediction", "result", "content", "model_output"):
        val = kwargs.get(key)
        if val is not None:
            return str(val)

    # single-dict style: grader({"output": "...", ...})
    if args and isinstance(args[0], dict):
        d = args[0]
        for key in ("output", "prediction", "result", "content", "model_output"):
            val = d.get(key)
            if val is not None:
                return str(val)

    # positional style: grader("output_value")
    if args and isinstance(args[0], str):
        return args[0]

    return ""


# ---------------------------------------------------------------------------
# Task 1: complaint classification
# ---------------------------------------------------------------------------

_TASK_1 = {
    "id": "email-task-001",
    "instruction": "Classify the email as complaint, query, or request.",
    "input": "I want a refund for my last order. The product was damaged.",
    "expected_output": "complaint",
    "benchmark": "email-triage-env",
    "score": 0.85,
}


def generated_task_1() -> dict[str, Any]:
    return dict(_TASK_1)


def generated_grader_1(*args: Any, **kwargs: Any) -> float:
    output = _extract_output(args, kwargs)
    if not output:
        return 0.5  # no output provided -> mid-range default
    return _clamp(_compare(output, _TASK_1["expected_output"]))


# ---------------------------------------------------------------------------
# Task 2: query classification
# ---------------------------------------------------------------------------

_TASK_2 = {
    "id": "email-task-002",
    "instruction": "Classify the email as complaint, query, or request.",
    "input": "Can you tell me the status of my order #12345?",
    "expected_output": "query",
    "benchmark": "email-triage-env",
    "score": 0.85,
}


def generated_task_2() -> dict[str, Any]:
    return dict(_TASK_2)


def generated_grader_2(*args: Any, **kwargs: Any) -> float:
    output = _extract_output(args, kwargs)
    if not output:
        return 0.5
    return _clamp(_compare(output, _TASK_2["expected_output"]))


# ---------------------------------------------------------------------------
# Task 3: request classification
# ---------------------------------------------------------------------------

_TASK_3 = {
    "id": "email-task-003",
    "instruction": "Classify the email as complaint, query, or request.",
    "input": "I'd like to request a demo of your enterprise plan.",
    "expected_output": "request",
    "benchmark": "email-triage-env",
    "score": 0.85,
}


def generated_task_3() -> dict[str, Any]:
    return dict(_TASK_3)


def generated_grader_3(*args: Any, **kwargs: Any) -> float:
    output = _extract_output(args, kwargs)
    if not output:
        return 0.5
    return _clamp(_compare(output, _TASK_3["expected_output"]))


# ---------------------------------------------------------------------------
# Task 4: another complaint
# ---------------------------------------------------------------------------

_TASK_4 = {
    "id": "email-task-004",
    "instruction": "Classify the email as complaint, query, or request.",
    "input": "This is unacceptable! I demand a refund immediately.",
    "expected_output": "complaint",
    "benchmark": "email-triage-env",
    "score": 0.85,
}


def generated_task_4() -> dict[str, Any]:
    return dict(_TASK_4)


def generated_grader_4(*args: Any, **kwargs: Any) -> float:
    output = _extract_output(args, kwargs)
    if not output:
        return 0.5
    return _clamp(_compare(output, _TASK_4["expected_output"]))


# ---------------------------------------------------------------------------
# Task 5: another query
# ---------------------------------------------------------------------------

_TASK_5 = {
    "id": "email-task-005",
    "instruction": "Classify the email as complaint, query, or request.",
    "input": "How do I reset my account password?",
    "expected_output": "query",
    "benchmark": "email-triage-env",
    "score": 0.85,
}


def generated_task_5() -> dict[str, Any]:
    return dict(_TASK_5)


def generated_grader_5(*args: Any, **kwargs: Any) -> float:
    output = _extract_output(args, kwargs)
    if not output:
        return 0.5
    return _clamp(_compare(output, _TASK_5["expected_output"]))


# ---------------------------------------------------------------------------
# Registry (for internal use)
# ---------------------------------------------------------------------------

TASKS = [
    {"task_fn": generated_task_1, "grader": generated_grader_1, "score": 0.85},
    {"task_fn": generated_task_2, "grader": generated_grader_2, "score": 0.85},
    {"task_fn": generated_task_3, "grader": generated_grader_3, "score": 0.85},
    {"task_fn": generated_task_4, "grader": generated_grader_4, "score": 0.85},
    {"task_fn": generated_task_5, "grader": generated_grader_5, "score": 0.85},
]

# Convenience aliases
easy_task = generated_task_1
medium_task = generated_task_2
hard_task = generated_task_3
easy_grader = generated_grader_1
medium_grader = generated_grader_2
hard_grader = generated_grader_3
