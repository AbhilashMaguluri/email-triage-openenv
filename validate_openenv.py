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


def _load_tasks_config() -> list[dict[str, str]]:
    with OPENENV_PATH.open(encoding="utf-8") as handle:
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
    return isinstance(score, float) and 0.0 < score < 1.0


def _extract_manifest_scores(task: dict[str, object]) -> list[float]:
    scores: list[float] = []

    direct_score = task.get("score")
    if isinstance(direct_score, (int, float)):
        scores.append(float(direct_score))

    scoring = task.get("scoring")
    if isinstance(scoring, dict):
        for value in scoring.values():
            if isinstance(value, (int, float)):
                scores.append(float(value))
    elif isinstance(scoring, (int, float)):
        scores.append(float(scoring))

    return scores


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _load_inference_module() -> object:
    previous = {
        "API_BASE_URL": os.environ.get("API_BASE_URL"),
        "MODEL_NAME": os.environ.get("MODEL_NAME"),
        "HF_TOKEN": os.environ.get("HF_TOKEN"),
    }
    try:
        os.environ.pop("API_BASE_URL", None)
        os.environ.pop("MODEL_NAME", None)
        os.environ["HF_TOKEN"] = "validator-token"

        spec = importlib.util.spec_from_file_location(
            "openenv_validator_inference",
            INFERENCE_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to create module spec for inference.py")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_env(previous)


def _validate_inference_source() -> None:
    if not INFERENCE_PATH.is_file():
        raise AssertionError("FAIL: inference.py must exist in the project root")

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
        raise AssertionError("FAIL: inference.py must import OpenAI from the openai package")
    if not constructs_client:
        raise AssertionError("FAIL: inference.py must construct an OpenAI client")

    module = _load_inference_module()
    api_base_url = getattr(module, "API_BASE_URL", None)
    model_name = getattr(module, "MODEL_NAME", None)
    hf_token = getattr(module, "HF_TOKEN", None)

    if not api_base_url:
        raise AssertionError("FAIL: API_BASE_URL must be read with a non-empty default")
    if not model_name:
        raise AssertionError("FAIL: MODEL_NAME must be read with a non-empty default")
    if hf_token != "validator-token":
        raise AssertionError("FAIL: HF_TOKEN must be read directly from the HF_TOKEN environment variable")


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
            "FAIL: inference.py exited with a non-zero status during smoke test\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not stdout_lines:
        raise AssertionError("FAIL: inference.py produced no stdout")

    if not START_RE.fullmatch(stdout_lines[0]):
        raise AssertionError(f"FAIL: invalid [START] line: {stdout_lines[0]}")
    if not END_RE.fullmatch(stdout_lines[-1]):
        raise AssertionError(f"FAIL: invalid [END] line: {stdout_lines[-1]}")

    step_lines = stdout_lines[1:-1]
    if not step_lines:
        raise AssertionError("FAIL: inference.py must emit at least one [STEP] line")

    step_numbers: list[int] = []
    for line in step_lines:
        match = STEP_RE.fullmatch(line)
        if not match:
            raise AssertionError(f"FAIL: invalid [STEP] line: {line}")
        step_numbers.append(int(match.group("step")))

    expected_steps = list(range(1, len(step_lines) + 1))
    if step_numbers != expected_steps:
        raise AssertionError(
            f"FAIL: [STEP] numbering must be sequential starting at 1, got {step_numbers}"
        )

    end_match = END_RE.fullmatch(stdout_lines[-1])
    assert end_match is not None
    end_steps = int(end_match.group("steps"))
    reward_values = [
        reward
        for reward in end_match.group("rewards").split(",")
        if reward != ""
    ]

    if end_steps != len(step_lines):
        raise AssertionError(
            f"FAIL: [END] steps={end_steps} does not match {len(step_lines)} [STEP] lines"
        )
    if len(reward_values) != len(step_lines):
        raise AssertionError(
            "FAIL: [END] rewards list length must match the number of [STEP] lines"
        )

    for line in stdout_lines:
        if not (
            START_RE.fullmatch(line)
            or STEP_RE.fullmatch(line)
            or END_RE.fullmatch(line)
        ):
            raise AssertionError(f"FAIL: unexpected stdout line: {line}")


def main() -> None:
    print("=" * 60)
    print("VALIDATION: OpenEnv Tasks, Graders, and Inference")
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
    print(f"    OK: {len(task_bindings)} task bindings found (>= 3)")

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

        manifest_scores = _extract_manifest_scores(task)
        if not manifest_scores:
            print("    FAIL: no manifest task score found")
            all_ok = False
            continue

        invalid_manifest_scores = [
            score for score in manifest_scores if not _validate_score(score)
        ]
        if invalid_manifest_scores:
            print(
                "    FAIL: manifest task scores must be floats strictly within (0, 1), "
                f"got {invalid_manifest_scores}"
            )
            all_ok = False
            continue
        print(f"    manifest scores={manifest_scores}")

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
            print(f"    OK: {score} is within (0, 1)")
        else:
            print(f"    FAIL: score must be float in (0, 1), got {score!r}")
            all_ok = False

    print("\n[4] Inference source validation:")
    try:
        _validate_inference_source()
        print("    OK: inference.py is in root, imports OpenAI, and reads required env vars")
    except Exception as exc:
        print(f"    FAIL: {exc}")
        all_ok = False

    print("\n[5] Inference runtime smoke test:")
    try:
        _validate_inference_runtime()
        print("    OK: inference.py emits valid [START]/[STEP]/[END] lines")
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


if __name__ == "__main__":
    main()
