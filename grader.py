from __future__ import annotations

import math
from typing import Any

MIN_OPEN_SCORE = 0.05
MAX_OPEN_SCORE = 0.95
EXACT_MATCH_SCORE = 0.95
PARTIAL_MATCH_SCORE = 0.75
TOKEN_OVERLAP_SCORE = 0.55
MISMATCH_SCORE = 0.15
EMPTY_EXPECTED_SCORE = 0.25


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
        return EMPTY_EXPECTED_SCORE
    if normalized_output == normalized_expected:
        return EXACT_MATCH_SCORE
    if normalized_output and (
        normalized_expected in normalized_output
        or normalized_output in normalized_expected
    ):
        return PARTIAL_MATCH_SCORE

    output_tokens = set(normalized_output.replace("/", " ").replace("_", " ").split())
    expected_tokens = set(
        normalized_expected.replace("/", " ").replace("_", " ").split()
    )
    if output_tokens and expected_tokens and output_tokens.intersection(expected_tokens):
        return TOKEN_OVERLAP_SCORE

    return MISMATCH_SCORE


def safe_grade(output: Any, expected: Any) -> float:
    try:
        score = float(grade(output=output, expected=expected))
    except Exception:
        return MIN_OPEN_SCORE

    if math.isnan(score) or math.isinf(score):
        return MIN_OPEN_SCORE
    if score <= 0.0:
        return MIN_OPEN_SCORE
    if score >= 1.0:
        return MAX_OPEN_SCORE

    return max(MIN_OPEN_SCORE, min(score, MAX_OPEN_SCORE))
