from __future__ import annotations

from tasks.tasks import TASKS


def main() -> None:
    for i, entry in enumerate(TASKS, 1):
        task = entry["task_fn"]()
        grader = entry["grader"]
        score = grader(output=task["expected_output"], expected=task["expected_output"])
        print(f"Task {i}: id={task['id']}  score={score}")


if __name__ == "__main__":
    main()
