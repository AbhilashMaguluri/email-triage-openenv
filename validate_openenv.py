from __future__ import annotations

import importlib
import sys

import yaml


def _load_tasks_config() -> list[dict[str, str]]:
    with open("openenv.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    tasks_config = config.get("tasks", [])
    if isinstance(tasks_config, dict):
        return [{"id": key, **value} for key, value in tasks_config.items()]
    if isinstance(tasks_config, list):
        return tasks_config
    raise TypeError("openenv.yaml field 'tasks' must be a list or dict")


def _load_task_bindings() -> list[dict[str, object]]:
    tasks_module = importlib.import_module("tasks.tasks")
    task_bindings = getattr(tasks_module, "TASKS", [])
    if not isinstance(task_bindings, list):
        raise TypeError("tasks.tasks:TASKS must be a list")
    return task_bindings


def _validate_score(score: object) -> bool:
    return isinstance(score, float) and 0.0 < score <= 1.0


def main() -> None:
    print("=" * 60)
    print("VALIDATION: OpenEnv Tasks & Graders")
    print("=" * 60)

    tasks_list = _load_tasks_config()
    print(f"\n[1] Tasks in openenv.yaml: {len(tasks_list)} found")
    if len(tasks_list) < 3:
        raise AssertionError(f"FAIL: need >= 3 tasks, got {len(tasks_list)}")
    print(f"    OK: {len(tasks_list)} tasks found (>= 3)")

    task_bindings = _load_task_bindings()
    print(f"\n[2] Dynamic TASKS bindings: {len(task_bindings)} found")
    if len(task_bindings) < 3:
        raise AssertionError(
            f"FAIL: need >= 3 task bindings, got {len(task_bindings)}"
        )

    all_ok = True
    for task in tasks_list:
        task_id = task.get("id", "unknown")
        entry_point = task.get("entry_point", "")
        grader_path = task.get("grader", "")

        print(f"\n[3] Task '{task_id}':")

        if not grader_path:
            print("    FAIL: no grader defined")
            all_ok = False
            continue

        task_state: dict[str, object]
        if entry_point:
            module_name, function_name = entry_point.rsplit(":", 1)
            module = importlib.import_module(module_name)
            task_callable = getattr(module, function_name)
            task_state = task_callable()
            print(f"    entry_point={entry_point} -> callable={callable(task_callable)}")
            print(f"    state type={type(task_state).__name__}")
        else:
            task_state = {}
            print("    entry_point=None (testing grader with empty state)")

        module_name, function_name = grader_path.rsplit(":", 1)
        module = importlib.import_module(module_name)
        grader_callable = getattr(module, function_name)
        score = grader_callable(task_state)
        print(f"    grader={grader_path} -> callable={callable(grader_callable)}")
        print(f"    score={score}")

        if _validate_score(score):
            print(f"    OK: {score} is within (0, 1]")
        else:
            print(f"    FAIL: score must be float in (0, 1], got {score!r}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
