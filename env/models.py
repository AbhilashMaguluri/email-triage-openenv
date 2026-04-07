from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    email: str
    category: Optional[str] = None
    priority: Optional[str] = None
    reply: Optional[str] = None


class Action(BaseModel):
    action_type: str = Field(
        ...,
        pattern=r"^(classify_email|set_priority|generate_reply)$",
    )
    content: Optional[str] = None


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
