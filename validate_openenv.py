"""
Local validator for OpenEnv submission.
Checks tasks, graders, scores, and inference output format.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
INFERENCE_PATH = ROOT / "inference.py"
OPENENV_PATH = ROOT / "openenv.yaml"

START_RE = re.compile(r"^\[START\] task=\S+ env=\S+ model=\S+$")
STEP_RE = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) action=(?P<action>.+) "
    r"reward=(?P<reward>-?\d+\.\d{2}) done=(?P<done>true|false) "
    r"error=(?P<error>.+)$"
)
END_RE = re.compile(
    r"^\[END\] success=(?P<success>true|false) "
    r"steps=(?P<steps>\d+) rewards=(?P<rewards>-?\d+\.\d{2}(?:,-?\d+\.\d{2})*|)$"
)


def _load_tasks_config() -> list[dict]:
    with OPENENV_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    tasks_config = config.get("tasks", [])
    if isinstance(tasks_config, list):
        return tasks_config
    raise TypeError("openenv.yaml field 'tasks' must be a list")


def _validate_score(score: object) -> bool:
    return isinstance(score, (int, float)) and 0.0 < float(score) < 1.0


def _get_scoring_value(task: dict) -> float | None:
    scoring = task.get("scoring")
    if isinstance(scoring, (int, float)):
        return float(scoring)
    if isinstance(scoring, dict):
        for val in scoring.values():
            if isinstance(val, (int, float)):
                return float(val)
    return None


def main() -> None:
    print("=" * 60)
    print("VALIDATION: OpenEnv Tasks, Graders, and Inference")
    print("=" * 60)

    tasks_list = _load_tasks_config()
    print(f"\n[1] Tasks in openenv.yaml: {len(tasks_list)} found")
    if len(tasks_list) < 3:
        raise AssertionError(f"FAIL: need >= 3 tasks, got {len(tasks_list)}")
    print(f"    OK: {len(tasks_list)} tasks found (>= 3)")

    all_ok = True
    for task in tasks_list:
        task_id = task.get("id", "unknown")
        entry_point = task.get("entry_point", "")
        grader_path = task.get("grader", "")

        print(f"\n[2] Task '{task_id}':")

        if not grader_path:
            print("    FAIL: no grader defined")
            all_ok = False
            continue

        scoring_val = _get_scoring_value(task)
        if scoring_val is None:
            print("    FAIL: no scoring value found")
            all_ok = False
            continue

        if not _validate_score(scoring_val):
            print(f"    FAIL: scoring value {scoring_val} not in (0, 1)")
            all_ok = False
            continue
        print(f"    scoring={scoring_val}")

        # Import and call the task entry point
        if entry_point:
            module_name, function_name = entry_point.rsplit(":", 1)
            module = importlib.import_module(module_name)
            task_callable = getattr(module, function_name)
            task_state = task_callable()
            print(f"    entry_point OK: type={type(task_state).__name__}")
        else:
            task_state = {}

        # Import and call the grader
        module_name, function_name = grader_path.rsplit(":", 1)
        module = importlib.import_module(module_name)
        grader_callable = getattr(module, function_name)

        # Test grader with task state (as dict positional arg)
        score1 = grader_callable(task_state)
        print(f"    grader(task_dict) = {score1}")

        # Test grader with no args
        score2 = grader_callable()
        print(f"    grader() = {score2}")

        # Test grader with kwargs
        score3 = grader_callable(
            output=task_state.get("expected_output", ""),
            expected=task_state.get("expected_output", ""),
        )
        print(f"    grader(output=expected, expected=expected) = {score3}")

        for label, score in [("task_dict", score1), ("no_args", score2), ("kwargs", score3)]:
            if _validate_score(score):
                print(f"    OK: {label} score {score} is within (0, 1)")
            else:
                print(f"    FAIL: {label} score {score!r} not in (0, 1)")
                all_ok = False

    print("\n[3] Inference source validation:")
    try:
        _validate_inference_source()
        print("    OK")
    except Exception as exc:
        print(f"    FAIL: {exc}")
        all_ok = False

    print("\n[4] Inference runtime smoke test:")
    try:
        _validate_inference_runtime()
        print("    OK")
    except Exception as exc:
        print(f"    FAIL: {exc}")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)
    print("=" * 60)


def _validate_inference_source() -> None:
    if not INFERENCE_PATH.is_file():
        raise AssertionError("inference.py must exist in the project root")

    source = INFERENCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(INFERENCE_PATH))

    imported_openai = False
    constructs_client = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "openai":
            imported_openai = any(alias.name == "OpenAI" for alias in node.names)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "OpenAI":
                constructs_client = True
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "OpenAI":
                constructs_client = True

    if not imported_openai:
        raise AssertionError("inference.py must import OpenAI from openai")
    if not constructs_client:
        raise AssertionError("inference.py must construct an OpenAI client")


def _validate_inference_runtime() -> None:
    env = os.environ.copy()
    env["API_BASE_URL"] = "http://127.0.0.1:9/v1"
    env["MODEL_NAME"] = "validator-model"
    env["HF_TOKEN"] = "validator-token"

    result = subprocess.run(
        [sys.executable, str(INFERENCE_PATH)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
        check=False,
    )

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0:
        raise AssertionError(
            f"inference.py exited non-zero\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not stdout_lines:
        raise AssertionError("inference.py produced no stdout")

    if not START_RE.fullmatch(stdout_lines[0]):
        raise AssertionError(f"invalid [START] line: {stdout_lines[0]}")
    if not END_RE.fullmatch(stdout_lines[-1]):
        raise AssertionError(f"invalid [END] line: {stdout_lines[-1]}")

    step_lines = stdout_lines[1:-1]
    if not step_lines:
        raise AssertionError("inference.py must emit at least one [STEP] line")

    for i, line in enumerate(step_lines, 1):
        match = STEP_RE.fullmatch(line)
        if not match:
            raise AssertionError(f"invalid [STEP] line: {line}")
        if int(match.group("step")) != i:
            raise AssertionError(f"step numbering wrong at line {i}")


if __name__ == "__main__":
    main()
