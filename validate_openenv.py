"""Validate that openenv.yaml tasks and graders are correctly detected."""
import yaml
import importlib
import sys

print("=" * 60)
print("VALIDATION: OpenEnv Tasks & Graders")
print("=" * 60)

# 1. Parse openenv.yaml
with open("openenv.yaml") as f:
    cfg = yaml.safe_load(f)

tasks_cfg = cfg.get("tasks", [])
print(f"\n[1] Tasks in openenv.yaml: {len(tasks_cfg)} found")

# Handle both list and dict formats
if isinstance(tasks_cfg, dict):
    tasks_list = [{"id": k, **v} for k, v in tasks_cfg.items()]
elif isinstance(tasks_cfg, list):
    tasks_list = tasks_cfg
else:
    print("FAIL: tasks must be a list or dict!")
    sys.exit(1)

assert len(tasks_list) >= 3, f"FAIL: need >= 3 tasks, got {len(tasks_list)}"
print(f"    OK: {len(tasks_list)} tasks found (>= 3)")

# 2. Verify each entry_point and grader is importable
all_ok = True
for task in tasks_list:
    name = task.get("id", "unknown")
    ep = task.get("entry_point", "")
    gr = task.get("grader", "")

    if not gr:
        print(f"\n[2] Task '{name}': FAIL - no grader defined")
        all_ok = False
        continue

    print(f"\n[2] Task '{name}':")

    # Import and call entry_point (if defined)
    if ep:
        mod_ep, fn_ep = ep.rsplit(":", 1)
        m1 = importlib.import_module(mod_ep)
        func_task = getattr(m1, fn_ep)
        state = func_task()
        print(f"    entry_point={ep} -> callable={callable(func_task)}")
        print(f"    state type={type(state).__name__}")
    else:
        state = {}
        print(f"    entry_point=None (testing grader with empty state)")

    # Import and call grader
    mod_gr, fn_gr = gr.rsplit(":", 1)
    m2 = importlib.import_module(mod_gr)
    func_grader = getattr(m2, fn_gr)

    # Test grader with the task state
    score = func_grader(state)
    print(f"    grader={gr} -> callable={callable(func_grader)}")
    print(f"    score={score}")

    # Validate score range — STRICTLY between 0 and 1
    if not isinstance(score, float):
        print(f"    FAIL: score must be float, got {type(score)}")
        all_ok = False
    elif score <= 0.0 or score >= 1.0:
        print(f"    FAIL: score {score} not in (0, 1)")
        all_ok = False
    else:
        print(f"    OK: {score} is strictly in (0, 1)")

print("\n" + "=" * 60)
if all_ok:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
print("=" * 60)
