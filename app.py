from fastapi import FastAPI
from pydantic import BaseModel

from env import EmailTriageEnv, Action

app = FastAPI(title="Email Triage API")
env = EmailTriageEnv()


class ActionRequest(BaseModel):
    action_type: str
    content: str = ""


@app.get("/")
def root():
    return {"message": "Email Triage API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/reset")
def reset():
    observation = env.reset()
    return observation.model_dump()


@app.post("/step")
def step(req: ActionRequest):
    action = Action(action_type=req.action_type, content=req.content)
    result = env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }
