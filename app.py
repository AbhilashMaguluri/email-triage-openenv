from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional

from env import EmailTriageEnv, Action

app = FastAPI(title="Email Triage API")
env = EmailTriageEnv()


class ActionRequest(BaseModel):
    action_type: str
    content: str = ""


# ── Root ──────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Email Triage API is running 🚀"}


# ── Health ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Reset (GET + POST) ───────────────────────────────────────

def _do_reset():
    """Shared reset logic – always returns HTTP 200 + observation JSON."""
    observation = env.reset()
    return observation.model_dump()


@app.get("/reset")
def reset_get():
    return _do_reset()


@app.post("/reset")
def reset_post():
    return _do_reset()


# ── Step (POST JSON body + GET query params) ──────────────────

def _do_step(action_type: str, content: str = ""):
    """Shared step logic – returns observation, reward, done, info."""
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
