from __future__ import annotations

import json

from task_generator import DEFAULT_TASK_COUNT, DEFAULT_TASK_SEED, generate_tasks
from tasks import model, run_task


def main() -> None:
    tasks = generate_tasks(n=DEFAULT_TASK_COUNT, seed=DEFAULT_TASK_SEED)
    print(f"number of tasks: {len(tasks)}")

    results = []
    for task in tasks:
        result = run_task(task)
        results.append(result)
        print(f"task id: {result['task']['id']}")
        print(f"score: {result['score']}")

    summary = {
        "task_count": len(tasks),
        "model_name": getattr(model, "__name__", "model"),
        "results": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
