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

tasks_cfg = cfg.get("tasks", {})
print(f"\n[1] Tasks in openenv.yaml: {list(tasks_cfg.keys())}")
assert isinstance(tasks_cfg, dict), "FAIL: tasks must be a mapping, not a list!"
assert len(tasks_cfg) >= 3, f"FAIL: need >= 3 tasks, got {len(tasks_cfg)}"
print("    OK: tasks is a dict with >= 3 entries")

# 2. Verify each entry_point and grader is importable
all_ok = True
for name, task in tasks_cfg.items():
    ep = task["entry_point"]
    gr = task["grader"]

    mod_ep, fn_ep = ep.rsplit(":", 1)
    mod_gr, fn_gr = gr.rsplit(":", 1)

    m1 = importlib.import_module(mod_ep)
    func_task = getattr(m1, fn_ep)

    m2 = importlib.import_module(mod_gr)
    func_grader = getattr(m2, fn_gr)

    # Run task to get state
    state = func_task()
    print(f"\n[2] Task '{name}':")
    print(f"    entry_point={ep} -> callable={callable(func_task)}")
    print(f"    state type={type(state).__name__}")

    # Run grader
    score = func_grader(state)
    print(f"    grader={gr} -> callable={callable(func_grader)}")
    print(f"    score={score}")

    # Validate score range
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
