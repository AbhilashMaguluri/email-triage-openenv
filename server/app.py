from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from env import EmailTriageEnv, Action, Observation
from env.environment import derive_email_expectations

README_PATH = Path(__file__).resolve().parent.parent / "README.md"

app = FastAPI(title="Email Triage Environment", version="0.1.0")
env = EmailTriageEnv()


# ── Request / Response Models ───────────────────────────────────

class ActionRequest(BaseModel):
    action_type: str
    content: str = ""


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class GraderRequest(BaseModel):
    episode_id: Optional[str] = None
    task_id: Optional[str] = None
    output: Optional[str] = None
    expected: Optional[str] = None


class TaskSummary(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int


class GraderResponse(BaseModel):
    task_id: str
    score: float = Field(..., gt=0.0, lt=1.0)
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)


# ── State model for /schema ─────────────────────────────────────

class EmailTriageState(BaseModel):
    email_id: str = ""
    observation: Dict[str, Any] = Field(default_factory=dict)
    step_index: int = 0
    total_reward: float = 0.0
    done: bool = True
    expected: Dict[str, str] = Field(default_factory=dict)
    last_action_error: Optional[str] = None


# ── Task Definitions ────────────────────────────────────────────

TASK_DEFS = [
    {
        "task_id": "email-task-001",
        "name": "Complaint Classification",
        "difficulty": "easy",
        "description": "Classify a complaint email about a damaged product.",
        "max_steps": 10,
        "input": "I want a refund for my last order. The product was damaged.",
        "expected_output": "complaint",
    },
    {
        "task_id": "email-task-002",
        "name": "Query Classification",
        "difficulty": "easy",
        "description": "Classify a query email about order status.",
        "max_steps": 10,
        "input": "Can you tell me the status of my order #12345?",
        "expected_output": "query",
    },
    {
        "task_id": "email-task-003",
        "name": "Request Classification",
        "difficulty": "medium",
        "description": "Classify a request email about an enterprise demo.",
        "max_steps": 10,
        "input": "I'd like to request a demo of your enterprise plan.",
        "expected_output": "request",
    },
    {
        "task_id": "email-task-004",
        "name": "Urgent Complaint Classification",
        "difficulty": "medium",
        "description": "Classify an urgent complaint demanding a refund.",
        "max_steps": 10,
        "input": "This is unacceptable! I demand a refund immediately.",
        "expected_output": "complaint",
    },
    {
        "task_id": "email-task-005",
        "name": "Password Query Classification",
        "difficulty": "hard",
        "description": "Classify a query about password reset.",
        "max_steps": 10,
        "input": "How do I reset my account password?",
        "expected_output": "query",
    },
]


def _grade(output: str, expected: str) -> float:
    """Grade output vs expected, returning score strictly in (0, 1)."""
    out = output.strip().lower()
    exp = expected.strip().lower()
    if out == exp:
        return 0.85
    if out and (exp in out or out in exp):
        return 0.65
    if out:
        return 0.15
    return 0.05


# ── Root ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "email-triage-env",
        "endpoints": [
            "/health", "/metadata", "/schema", "/tasks",
            "/reset", "/step", "/state", "/grader", "/mcp",
        ],
    }


# ── Health ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "service": "email-triage-env"}


# ── MCP (JSON-RPC stub) ────────────────────────────────────────

@app.post("/mcp")
def mcp(request: dict = None):
    req = request or {}
    return {
        "jsonrpc": "2.0",
        "id": req.get("id", 1),
        "result": {
            "tools": [],
            "description": "email-triage-env MCP endpoint",
        },
    }


# ── Metadata ────────────────────────────────────────────────────

@app.get("/metadata")
def metadata():
    readme_content = None
    if README_PATH.exists():
        readme_content = README_PATH.read_text(encoding="utf-8")
    return {
        "name": "email-triage-env",
        "description": "AI Email Triage & Response Environment that classifies emails, assigns priority, and generates replies.",
        "version": app.version,
        "readme_content": readme_content,
    }


# ── Schema ──────────────────────────────────────────────────────

@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EmailTriageState.model_json_schema(),
    }


# ── Tasks ───────────────────────────────────────────────────────

@app.get("/tasks")
def tasks():
    return [
        TaskSummary(
            task_id=t["task_id"],
            name=t["name"],
            difficulty=t["difficulty"],
            description=t["description"],
            max_steps=t["max_steps"],
        ).model_dump(mode="json")
        for t in TASK_DEFS
    ]


# ── Grader ──────────────────────────────────────────────────────

@app.post("/grader")
def grader(payload: GraderRequest):
    if payload.task_id is not None:
        task_def = None
        for t in TASK_DEFS:
            if t["task_id"] == payload.task_id:
                task_def = t
                break
        if task_def is None:
            raise HTTPException(status_code=404, detail=f"Unknown task: {payload.task_id}")

        output = payload.output or ""
        expected = payload.expected or task_def["expected_output"]
        score = _grade(output, expected)

        return GraderResponse(
            task_id=payload.task_id,
            score=score,
            passed=score >= 0.7,
            details={
                "output": output,
                "expected": expected,
            },
        ).model_dump(mode="json")

    raise HTTPException(status_code=400, detail="Provide task_id")


# ── Reset ───────────────────────────────────────────────────────

def _do_reset(task_id: str | None = None):
    if task_id:
        task_def = None
        for t in TASK_DEFS:
            if t["task_id"] == task_id:
                task_def = t
                break
        if task_def:
            env._current_email = {"id": task_def["task_id"], "body": task_def["input"]}
            env._expected = derive_email_expectations(task_def["input"])
            env._observation = Observation(email=task_def["input"])
            env._step_index = 0
            env._total_reward = 0.0
            env._done = False
            env.last_action_error = None
            return env._observation.model_dump()

    observation = env.reset()
    return observation.model_dump()


@app.get("/reset")
def reset_get(task_id: str | None = Query(default=None)):
    return _do_reset(task_id)


@app.post("/reset")
def reset_post(req: ResetRequest = None):
    task_id = req.task_id if req else None
    return _do_reset(task_id)


# ── Step ────────────────────────────────────────────────────────

def _do_step(action_type: str, content: str = ""):
    action = Action(action_type=action_type, content=content)
    result = env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.post("/step")
def step_post(req: ActionRequest):
    return _do_step(action_type=req.action_type, content=req.content)


@app.get("/step")
def step_get(
    action_type: str = Query(..., description="Action type"),
    content: str = Query("", description="Action content"),
):
    return _do_step(action_type=action_type, content=content)


# ── State ───────────────────────────────────────────────────────

@app.get("/state")
def state():
    return env.state()


# ── CLI Entry Point ─────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
