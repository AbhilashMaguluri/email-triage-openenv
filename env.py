from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable

from grader import safe_grade
from task_generator import DEFAULT_TASK_COUNT, DEFAULT_TASK_SEED, generate_tasks

_PACKAGE_DIR = Path(__file__).resolve().parent / "env"

if __name__ != "__main__" and _PACKAGE_DIR.is_dir():
    __path__ = [str(_PACKAGE_DIR)]
    if __spec__ is not None:
        __spec__.submodule_search_locations = list(__path__)

    try:
        _environment_module = importlib.import_module("env.environment")
        _models_module = importlib.import_module("env.models")
        EmailTriageEnv = _environment_module.EmailTriageEnv
        Action = _models_module.Action
        Observation = _models_module.Observation
        StepResult = _models_module.StepResult
    except Exception:
        EmailTriageEnv = None
        Action = None
        Observation = None
        StepResult = None


def model(input_text: str | None) -> str:
    normalized = " ".join(str(input_text or "").strip().lower().split())

    if not normalized:
        return "normal"
    if "free" in normalized or "offer" in normalized:
        return "spam"
    if "bank" in normalized or "password" in normalized:
        return "important"
    return "normal"


def _coerce_task(task: dict[str, Any]) -> dict[str, str]:
    task_id = str(task.get("id") or "").strip() or "email-task-unknown"
    task_input = str(task.get("input") or "").strip()
    expected_output = str(task.get("expected_output") or "").strip().lower() or "normal"
    return {
        "id": task_id,
        "input": task_input,
        "expected_output": expected_output,
    }


def run_task(
    task: dict[str, Any],
    model_fn: Callable[[str | None], Any] | None = None,
) -> dict[str, Any]:
    task_data = _coerce_task(task)
    active_model = model_fn or model

    try:
        raw_output = active_model(task_data["input"])
    except Exception as exc:
        raw_output = ""
        model_error = str(exc)
    else:
        model_error = ""

    model_output = "" if raw_output is None else str(raw_output).strip()
    score = safe_grade(output=model_output, expected=task_data["expected_output"])

    return {
        "task_id": task_data["id"],
        "input": task_data["input"],
        "expected_output": task_data["expected_output"],
        "model_output": model_output,
        "score": score,
        "passed": score >= 0.7,
        "error": model_error,
    }


def run_all_tasks(
    n: int = DEFAULT_TASK_COUNT,
    seed: int | None = DEFAULT_TASK_SEED,
    model_fn: Callable[[str | None], Any] | None = None,
) -> dict[str, Any]:
    tasks = generate_tasks(n=n, seed=seed)
    if not tasks:
        raise ValueError("No tasks available for execution.")
    if len(tasks) < 3:
        raise ValueError(f"Expected at least 3 tasks, received {len(tasks)}")

    results = [run_task(task=task, model_fn=model_fn) for task in tasks]
    average_score = sum(result["score"] for result in results) / len(results)

    return {
        "task_count": len(tasks),
        "average_score": round(average_score, 4),
        "results": results,
    }


def main() -> None:
    summary = run_all_tasks()
    print(f"number of tasks: {summary['task_count']}")
    for result in summary["results"]:
        print(f"task id: {result['task_id']}")
        print(f"score: {result['score']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
