from __future__ import annotations

import random
from typing import Final

DEFAULT_TASK_COUNT: Final[int] = 5
DEFAULT_TASK_SEED: Final[int] = 20260410

_EMAIL_SAMPLE_BANK: tuple[dict[str, str], ...] = (
    {
        "category": "spam",
        "input": "Subject: Free vacation offer just for you. Claim your free reward before midnight and unlock our best offer.",
    },
    {
        "category": "spam",
        "input": "Subject: Exclusive offer. You have been selected for a free shopping voucher if you reply in the next hour.",
    },
    {
        "category": "spam",
        "input": "Subject: Mega offer from our partner network. Get free access to premium tools when you click today.",
    },
    {
        "category": "important",
        "input": "Subject: Bank security notice. We detected unusual activity on your bank account and need you to review it immediately.",
    },
    {
        "category": "important",
        "input": "Subject: Password reset confirmation. Your password was changed recently, so please verify this action if it was not you.",
    },
    {
        "category": "important",
        "input": "Subject: Urgent payroll issue. The finance team needs you to update your bank details before today's processing window closes.",
    },
    {
        "category": "normal",
        "input": "Subject: Team lunch tomorrow. Please confirm whether you can join the product team at noon.",
    },
    {
        "category": "normal",
        "input": "Subject: Weekly planning notes. Sharing the agenda for Monday's design review and next sprint kickoff.",
    },
    {
        "category": "normal",
        "input": "Subject: Shipping update. Your package is on the way and should arrive later this week.",
    },
)


def classify_email_text(text: str | None) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())

    if not normalized:
        return "normal"
    if "free" in normalized or "offer" in normalized:
        return "spam"
    if "bank" in normalized or "password" in normalized:
        return "important"
    return "normal"


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


def _build_task(sample: dict[str, str], index: int, cycle: int) -> dict[str, str]:
    task_input = _build_unique_input(sample["input"], index, cycle)
    expected_output = classify_email_text(task_input)

    return {
        "id": f"email-task-{index:03d}",
        "input": task_input,
        "expected_output": expected_output,
    }


def generate_tasks(
    n: int = DEFAULT_TASK_COUNT,
    seed: int | None = DEFAULT_TASK_SEED,
) -> list[dict[str, str]]:
    requested = _coerce_task_count(n)
    rng = random.Random(_coerce_seed(seed))
    sample_pool = list(_EMAIL_SAMPLE_BANK)
    rng.shuffle(sample_pool)

    generated: list[dict[str, str]] = []
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
