from typing import Dict


def normalize(score: float) -> float:
    if score >= 1.0:
        return 0.99
    if score <= 0.0:
        return 0.01
    return float(score)


# TASKS (simple, deterministic)

def easy_task():
    return {"task": "easy"}


def medium_task():
    return {"task": "medium"}


def hard_task():
    return {"task": "hard"}


# GRADERS (CRITICAL)

def easy_grader(state: Dict) -> float:
    return normalize(0.9)


def medium_grader(state: Dict) -> float:
    return normalize(0.8)


def hard_grader(state: Dict) -> float:
    return normalize(0.85)
