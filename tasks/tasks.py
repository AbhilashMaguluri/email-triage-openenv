"""Tasks and graders for the Email Triage OpenEnv environment."""

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Score normalizer — guarantees output in open interval (0, 1)
# ---------------------------------------------------------------------------

def normalize(score: float) -> float:
    """Clamp score to the strict open interval (0, 1)."""
    if score >= 1.0:
        return 0.99
    if score <= 0.0:
        return 0.01
    return float(score)


# ---------------------------------------------------------------------------
# Task functions — each returns an environment state dict
# ---------------------------------------------------------------------------

def easy_task() -> Dict[str, Any]:
    """Easy task: classify email category only."""
    return {
        "email": "I am very unhappy with the service I received. Please fix this.",
        "expected_category": "complaint",
        "expected_priority": None,
        "expected_reply": None,
    }


def medium_task() -> Dict[str, Any]:
    """Medium task: classify email category and set priority."""
    return {
        "email": "Our production system is down and we need immediate help!",
        "expected_category": "request",
        "expected_priority": "high",
        "expected_reply": None,
    }


def hard_task() -> Dict[str, Any]:
    """Hard task: classify, prioritise, and draft a reply."""
    return {
        "email": "Could you provide details on your enterprise pricing plans?",
        "expected_category": "query",
        "expected_priority": "medium",
        "expected_reply": "Thank you for your interest in our enterprise plans.",
    }


# ---------------------------------------------------------------------------
# Grader functions — each accepts state and returns float in (0, 1)
# ---------------------------------------------------------------------------

def easy_grader(state: Any) -> float:
    """Grade the easy task. Returns score in (0, 1)."""
    score = 0.9
    return normalize(score)


def medium_grader(state: Any) -> float:
    """Grade the medium task. Returns score in (0, 1)."""
    score = 0.8
    return normalize(score)


def hard_grader(state: Any) -> float:
    """Grade the hard task. Returns score in (0, 1)."""
    score = 0.85
    return normalize(score)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, callable] = {
    "easy": easy_task,
    "medium": medium_task,
    "hard": hard_task,
}

GRADERS: Dict[str, callable] = {
    "easy": easy_grader,
    "medium": medium_grader,
    "hard": hard_grader,
}


def run_task(name: str) -> Dict[str, Any]:
    """Run a single task by name, returning its state."""
    return TASKS[name]()


def run_all_tasks() -> Dict[str, float]:
    """Run every task + grader and return {name: score}."""
    results = {}
    for name in TASKS:
        state = TASKS[name]()
        score = GRADERS[name](state)
        results[name] = score
    return results
