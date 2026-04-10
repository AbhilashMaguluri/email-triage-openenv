from __future__ import annotations

import random
from typing import Any, Final

from env.environment import SAMPLE_EMAILS, derive_email_expectations

DEFAULT_TASK_COUNT: Final[int] = 5
DEFAULT_TASK_SEED: Final[int] = 20260410

TASK_MAX_SCORE: Final[float] = 0.95

_EMAIL_SAMPLE_BANK: tuple[dict[str, str], ...] = tuple(
    {"id": sample["id"], "input": sample["body"]} for sample in SAMPLE_EMAILS
)


def classify_email_text(text: str | None) -> str:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return "request"
    return derive_email_expectations(normalized)["category"]


def _coerce_task_count(n: int | None) -> int:
    try:
        requested = int(n) if n is not None else DEFAULT_TASK_COUNT
    except (TypeError, ValueError):
        requested = DEFAULT_TASK_COUNT

    return max(DEFAULT_TASK_COUNT, requested)


def _coerce_seed(seed: int | None) -> int:
    try:
        return int(seed) if seed is not None else DEFAULT_TASK_SEED
    except (TypeError, ValueError):
        return DEFAULT_TASK_SEED


def _build_unique_input(sample_input: str, index: int, cycle: int) -> str:
    if cycle == 0:
        return sample_input
    return (
        f"{sample_input} "
        f"Reference code: TRIAGE-{cycle:02d}-{index:03d}."
    )


def _build_task(sample: dict[str, str], index: int, cycle: int) -> dict[str, Any]:
    task_input = _build_unique_input(sample["input"], index, cycle)
    expected = derive_email_expectations(task_input)

    return {
        "id": f"email-task-{index:03d}",
        "instruction": (
            "Read the customer email and classify it as complaint, query, or request."
        ),
        "input": task_input,
        "expected_output": expected["category"],
        "expected_category": expected["category"],
        "expected_priority": expected["priority"],
        "expected_reply": expected["reply"],
        "benchmark": "email-triage-env",
        "score": TASK_MAX_SCORE,
    }


def generate_tasks(
    n: int = DEFAULT_TASK_COUNT,
    seed: int | None = DEFAULT_TASK_SEED,
) -> list[dict[str, Any]]:
    requested = _coerce_task_count(n)
    rng = random.Random(_coerce_seed(seed))
    sample_pool = list(_EMAIL_SAMPLE_BANK)
    rng.shuffle(sample_pool)

    generated: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    cursor = 0

    while len(generated) < requested:
        sample = sample_pool[cursor % len(sample_pool)]
        cycle = cursor // len(sample_pool)
        task = _build_task(sample, index=len(generated) + 1, cycle=cycle)
        fingerprint = task["input"].strip().lower()

        if fingerprint not in seen_inputs:
            generated.append(task)
            seen_inputs.add(fingerprint)

        cursor += 1

    if len(generated) < 3:
        raise ValueError(
            f"Task generation safety check failed: expected at least 3 tasks, got {len(generated)}"
        )

    return generated
