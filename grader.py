from __future__ import annotations

import math
from typing import Any


def _extract_value(value: Any) -> Any:
    if value is None:
        return ""

    if isinstance(value, dict):
        for key in (
            "output",
            "expected_output",
            "expected",
            "prediction",
            "label",
            "result",
            "content",
            "value",
        ):
            candidate = value.get(key)
            if candidate is not None:
                return candidate
        return ""

    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value if item is not None)

    return value


def normalize_label(value: Any) -> str:
    extracted = _extract_value(value)
    return " ".join(str(extracted or "").strip().lower().split())


def grade(output: Any, expected: Any) -> float:
    normalized_output = normalize_label(output)
    normalized_expected = normalize_label(expected)

    if not normalized_expected:
        return 0.2
    if normalized_output == normalized_expected:
        return 1.0
    if normalized_output and (
        normalized_expected in normalized_output
        or normalized_output in normalized_expected
    ):
        return 0.7

    output_tokens = set(normalized_output.replace("/", " ").replace("_", " ").split())
    expected_tokens = set(
        normalized_expected.replace("/", " ").replace("_", " ").split()
    )
    if output_tokens and expected_tokens and output_tokens.intersection(expected_tokens):
        return 0.7

    return 0.2


def safe_grade(output: Any, expected: Any) -> float:
    try:
        score = float(grade(output=output, expected=expected))
    except Exception:
        return 0.1

    if math.isnan(score) or math.isinf(score):
        return 0.1
    if score <= 0.0 or score > 1.0:
        return 0.1

    return score
