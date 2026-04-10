from __future__ import annotations

from typing import Any, Callable

from grader import safe_grade
from task_generator import (
    DEFAULT_TASK_COUNT,
    DEFAULT_TASK_SEED,
    generate_tasks as _generate_tasks,
)

TaskDict = dict[str, str]
GraderFn = Callable[..., float]


class TaskObject(dict):
    def __call__(self) -> dict[str, str]:
        return dict(self)


def generate_tasks(
    n: int = DEFAULT_TASK_COUNT,
    seed: int | None = DEFAULT_TASK_SEED,
) -> list[TaskDict]:
    tasks = _generate_tasks(n=n, seed=seed)
    if len(tasks) < 3:
        raise ValueError(
            f"Not enough tasks with graders: generated {len(tasks)} tasks, expected at least 3"
        )
    return tasks


def _clone_task(task: TaskDict) -> TaskDict:
    return TaskObject(
        {
            "id": str(task["id"]),
            "input": str(task["input"]),
            "expected_output": str(task["expected_output"]),
        }
    )


def _default_task_bank() -> list[TaskDict]:
    return [_clone_task(task) for task in generate_tasks()]


def _extract_info(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    info.update(kwargs)

    for value in args:
        if isinstance(value, dict):
            info.update(value)
        elif isinstance(value, str) and "output" not in info:
            info["output"] = value
        elif value is not None and "output" not in info:
            info["output"] = value

    return info


def _resolve_output_expected(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    fallback_expected: str | None = None,
) -> tuple[Any, Any]:
    info = _extract_info(args, kwargs)

    output = None
    for key in ("output", "prediction", "result", "label", "content", "model_output"):
        if info.get(key) is not None:
            output = info.get(key)
            break

    expected = None
    for key in ("expected", "expected_output", "target", "label_expected"):
        if info.get(key) is not None:
            expected = info.get(key)
            break

    if expected is None:
        expected = fallback_expected
    if output is None and expected is not None:
        output = expected

    return output, expected


def universal_grader(*args: Any, **kwargs: Any) -> float:
    output, expected = _resolve_output_expected(args=args, kwargs=kwargs)
    return safe_grade(output=output, expected=expected)


DEFAULT_TASK_OBJECTS: list[TaskDict] = _default_task_bank()


def _build_task_grader(task: TaskDict) -> GraderFn:
    expected_output = task["expected_output"]

    def _grader(*args: Any, **kwargs: Any) -> float:
        output, expected = _resolve_output_expected(
            args=args,
            kwargs=kwargs,
            fallback_expected=expected_output,
        )
        return safe_grade(output=output, expected=expected)

    return _grader


TASKS: list[dict[str, Any]] = []
TASK_REGISTRY: dict[str, dict[str, Any]] = {}
ENTRYPOINT_NAMES: list[str] = []

for index, task in enumerate(DEFAULT_TASK_OBJECTS, start=1):
    task_copy = _clone_task(task)
    grader = _build_task_grader(task_copy)
    task_name = f"generated_task_{index}"
    grader_name = f"generated_grader_{index}"

    def _make_task_loader(task_snapshot: TaskDict) -> Callable[[], TaskDict]:
        def _loader() -> TaskDict:
            return _clone_task(task_snapshot)

        return _loader

    task_loader = _make_task_loader(task_copy)
    task_loader.__name__ = task_name
    grader.__name__ = grader_name

    globals()[task_name] = task_loader
    globals()[grader_name] = grader

    ENTRYPOINT_NAMES.append(task_name)
    TASKS.append(
        {
            "task": task_loader(),
            "task_fn": task_loader,
            "grader": grader,
        }
    )
    TASK_REGISTRY[task_copy["id"]] = {"task": task_loader, "grader": grader}


def _task_from_index(index: int) -> TaskDict:
    if index < 1 or index > len(DEFAULT_TASK_OBJECTS):
        raise IndexError(
            f"Task index out of range: {index}. Available range: 1..{len(DEFAULT_TASK_OBJECTS)}"
        )
    return _clone_task(DEFAULT_TASK_OBJECTS[index - 1])


def list_tasks() -> list[TaskDict]:
    return [_clone_task(task) for task in DEFAULT_TASK_OBJECTS]


def get_task(name: str | int) -> TaskDict:
    if isinstance(name, int):
        return _task_from_index(name)

    task_name = str(name).strip().lower()
    alias_map = {
        "easy": 1,
        "medium": 2,
        "hard": 3,
    }
    if task_name in alias_map:
        return _task_from_index(alias_map[task_name])

    for task in DEFAULT_TASK_OBJECTS:
        if task["id"] == task_name:
            return _clone_task(task)

    raise KeyError(
        f"Unknown task: {name}. Available ids: {[task['id'] for task in DEFAULT_TASK_OBJECTS]}"
    )


def get_grader(name: str | int) -> GraderFn:
    if isinstance(name, int):
        task = _task_from_index(name)
        return TASK_REGISTRY[task["id"]]["grader"]

    task_name = str(name).strip().lower()
    alias_map = {
        "easy": 1,
        "medium": 2,
        "hard": 3,
    }
    if task_name in alias_map:
        task = _task_from_index(alias_map[task_name])
        return TASK_REGISTRY[task["id"]]["grader"]

    if task_name in TASK_REGISTRY:
        return TASK_REGISTRY[task_name]["grader"]

    raise KeyError(
        f"Unknown grader for task: {name}. Available ids: {list(TASK_REGISTRY.keys())}"
    )


def model(input_text: str | None) -> str:
    normalized = " ".join(str(input_text or "").strip().lower().split())

    if not normalized:
        return "normal"
    if "free" in normalized or "offer" in normalized:
        return "spam"
    if "bank" in normalized or "password" in normalized:
        return "important"
    return "normal"


def run_task(task_name: str | int | TaskDict) -> dict[str, Any]:
    task = get_task(task_name) if not isinstance(task_name, dict) else _clone_task(task_name)
    grader = get_grader(task["id"])

    try:
        output = model(task["input"])
    except Exception:
        output = ""

    score = grader(output=output, expected_output=task["expected_output"])
    return {
        "task": task,
        "output": output,
        "score": score,
        "passed": score >= 0.7,
    }


def run_all_tasks() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for task in DEFAULT_TASK_OBJECTS:
        result = run_task(task)
        results[task["id"]] = result
    return results


easy_task = generated_task_1
medium_task = generated_task_2
hard_task = generated_task_3
easy_grader = generated_grader_1
medium_grader = generated_grader_2
hard_grader = generated_grader_3
