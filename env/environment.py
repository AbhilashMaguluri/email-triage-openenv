from __future__ import annotations

import random
from typing import Any

from .models import Action, Observation, StepResult

SAMPLE_EMAILS: list[dict[str, str]] = [
    {
        "body": "I want a refund for my last order. The product was damaged.",
        "expected_category": "complaint",
        "expected_priority": "high",
        "expected_reply": "We are processing your refund.",
    },
    {
        "body": "Can you tell me the status of my order #12345?",
        "expected_category": "query",
        "expected_priority": "medium",
        "expected_reply": "Your order is being prepared and will ship soon.",
    },
    {
        "body": "I'd like to request a demo of your enterprise plan.",
        "expected_category": "request",
        "expected_priority": "low",
        "expected_reply": "We'd be happy to schedule a demo for you.",
    },
    {
        "body": "This is unacceptable! I demand a refund immediately.",
        "expected_category": "complaint",
        "expected_priority": "high",
        "expected_reply": "We are processing your refund.",
    },
    {
        "body": "How do I reset my account password?",
        "expected_category": "query",
        "expected_priority": "medium",
        "expected_reply": "You can reset your password from the login page.",
    },
    {
        "body": "Please add me to your newsletter mailing list.",
        "expected_category": "request",
        "expected_priority": "low",
        "expected_reply": "You have been added to our mailing list.",
    },
]

ACTION_SEQUENCE = ["classify_email", "set_priority", "generate_reply"]


def _derive_expected(email_body: str) -> dict[str, str]:
    body_lower = email_body.lower()

    if "refund" in body_lower or "unacceptable" in body_lower or "damaged" in body_lower:
        category = "complaint"
    elif any(kw in body_lower for kw in ("status", "how", "where", "when", "password")):
        category = "query"
    else:
        category = "request"

    if category == "complaint":
        priority = "high"
    elif category == "query":
        priority = "medium"
    else:
        priority = "low"

    if priority == "high":
        reply = "We are processing your refund."
    elif priority == "medium":
        reply = "Your query has been noted and we will respond shortly."
    else:
        reply = "We have received your request and will follow up."

    return {"category": category, "priority": priority, "reply": reply}


class EmailTriageEnv:
    def __init__(self, emails: list[dict[str, str]] | None = None) -> None:
        self._emails = emails or SAMPLE_EMAILS
        self._current_email: dict[str, str] = {}
        self._observation: Observation = Observation(email="")
        self._step_index: int = 0
        self._total_reward: float = 0.0
        self._done: bool = True
        self._expected: dict[str, str] = {}
        self.last_action_error: str | None = None

    def reset(self) -> Observation:
        email_data = random.choice(self._emails)
        self._current_email = email_data
        self._expected = _derive_expected(email_data["body"])
        self._observation = Observation(email=email_data["body"])
        self._step_index = 0
        self._total_reward = 0.0
        self._done = False
        self.last_action_error = None
        return self._observation.model_copy()

    def step(self, action: Action) -> StepResult:
        if self._done:
            self.last_action_error = "Episode is done. Call reset() before stepping."
            raise RuntimeError("Episode is done. Call reset() before stepping.")

        expected_action = ACTION_SEQUENCE[self._step_index]
        reward = 0.0
        info: dict[str, Any] = {"step": self._step_index + 1}
        self.last_action_error = None

        if action.action_type != expected_action:
            reward = -0.2
            info["error"] = f"Expected '{expected_action}', got '{action.action_type}'"
            self.last_action_error = info["error"]
            self._total_reward += reward
            return StepResult(
                observation=self._observation.model_copy(),
                reward=reward,
                done=False,
                info=info,
            )

        if action.action_type == "classify_email":
            reward, info = self._handle_classify(action, info)
        elif action.action_type == "set_priority":
            reward, info = self._handle_priority(action, info)
        elif action.action_type == "generate_reply":
            reward, info = self._handle_reply(action, info)

        self._total_reward += reward
        self._step_index += 1

        if self._step_index >= len(ACTION_SEQUENCE):
            self._done = True
            info["total_reward"] = round(self._total_reward, 2)

        return StepResult(
            observation=self._observation.model_copy(),
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> dict[str, Any]:
        return {
            "observation": self._observation.model_dump(),
            "step_index": self._step_index,
            "total_reward": round(self._total_reward, 2),
            "done": self._done,
            "expected": self._expected,
            "last_action_error": self.last_action_error,
        }

    def close(self) -> None:
        self._done = True

    def _handle_classify(
        self, action: Action, info: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        expected = self._expected["category"]
        provided = (action.content or "").strip().lower()

        if provided == expected:
            self._observation.category = provided
            info["match"] = True
            self.last_action_error = None
            return 0.4, info

        self._observation.category = provided
        info["match"] = False
        info["expected"] = expected
        info["error"] = f"Expected category '{expected}', got '{provided}'"
        self.last_action_error = info["error"]
        return -0.2, info

    def _handle_priority(
        self, action: Action, info: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        expected = self._expected["priority"]
        provided = (action.content or "").strip().lower()

        if provided == expected:
            self._observation.priority = provided
            info["match"] = True
            self.last_action_error = None
            return 0.3, info

        self._observation.priority = provided
        info["match"] = False
        info["expected"] = expected
        info["error"] = f"Expected priority '{expected}', got '{provided}'"
        self.last_action_error = info["error"]
        return -0.2, info

    def _handle_reply(
        self, action: Action, info: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        expected = self._expected["reply"]
        provided = (action.content or "").strip()

        if provided == expected:
            self._observation.reply = provided
            info["match"] = True
            self.last_action_error = None
            return 0.3, info

        self._observation.reply = provided
        info["match"] = False
        info["expected"] = expected
        info["error"] = f"Expected reply '{expected}', got '{provided}'"
        self.last_action_error = info["error"]
        return -0.2, info
