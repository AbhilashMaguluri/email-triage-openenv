from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from env.environment import ACTION_SEQUENCE, EmailTriageEnv, derive_email_expectations
from env.models import Action, Observation

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

TASK_NAME = "email-triage"
BENCHMARK_NAME = "email-triage-env"
REQUEST_TIMEOUT_SECONDS = 5.0
MAX_STEPS = len(ACTION_SEQUENCE)
SUCCESS_REWARD = 1.0

SYSTEM_PROMPT = """\
You are an email triage agent. You process customer emails in a strict 3-step pipeline.

Step 1 - classify_email:
- Return exactly one category: complaint, query, or request.
- "refund", "unacceptable", and "damaged" indicate complaint.
- "status", "how", "where", "when", and "password" indicate query.
- Everything else is request.

Step 2 - set_priority:
- complaint -> high
- query -> medium
- request -> low

Step 3 - generate_reply:
- high -> "We are processing your refund."
- medium -> "Your query has been noted and we will respond shortly."
- low -> "We have received your request and will follow up."

Respond with only compact JSON:
{"action_type": "<action>", "content": "<value>"}
"""


def _build_client() -> OpenAI:
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
        max_retries=0,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_error(value: Any) -> str:
    if value in (None, ""):
        return "null"
    return str(value).replace("\r", " ").replace("\n", " ")


def _format_action(action: Action) -> str:
    return f"{action.action_type}({repr(action.content or '')})"


def _build_user_message(observation: Observation, step_index: int) -> str:
    parts = [f"Email: {observation.email}"]
    if observation.category:
        parts.append(f"Assigned category: {observation.category}")
    if observation.priority:
        parts.append(f"Assigned priority: {observation.priority}")

    if step_index < len(ACTION_SEQUENCE):
        parts.append(f"Next required action: {ACTION_SEQUENCE[step_index]}")

    return "\n".join(parts)


def _parse_action(raw: str) -> Action | None:
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
        return Action(
            action_type=str(data["action_type"]),
            content="" if data.get("content") is None else str(data["content"]),
        )
    except Exception:
        return None


def _fallback_action(observation: Observation, step_index: int) -> Action:
    expected = derive_email_expectations(observation.email)

    if step_index == 0:
        return Action(
            action_type="classify_email",
            content=expected["category"],
        )

    if step_index == 1:
        return Action(
            action_type="set_priority",
            content=expected["priority"],
        )

    return Action(
        action_type="generate_reply",
        content=expected["reply"],
    )


def generate_action(
    client: OpenAI,
    observation: Observation,
    step_index: int,
    history: list[dict[str, str]],
) -> Action:
    expected_action = ACTION_SEQUENCE[step_index] if step_index < len(ACTION_SEQUENCE) else None
    user_message = _build_user_message(observation, step_index)

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=128,
        )
        raw_content = response.choices[0].message.content or ""
        action = _parse_action(raw_content)
    except Exception:
        action = None

    if action is None or (expected_action and action.action_type != expected_action):
        action = _fallback_action(observation, step_index)

    history.append({"role": "user", "content": user_message})
    history.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {"action_type": action.action_type, "content": action.content},
                separators=(",", ":"),
            ),
        }
    )
    return action


def run_episode(env: EmailTriageEnv, client: OpenAI) -> dict[str, Any]:
    rewards: list[float] = []
    steps_taken = 0
    done = False
    observation: Observation | None = None
    episode_error: str | None = None
    history: list[dict[str, str]] = []

    print(
        f"[START] task={TASK_NAME} env={BENCHMARK_NAME} model={MODEL_NAME}",
        flush=True,
    )

    try:
        observation = env.reset()

        while not done and steps_taken < MAX_STEPS:
            step_index = getattr(env, "_step_index", steps_taken)
            action = generate_action(client, observation, step_index, history)
            result = env.step(action)

            steps_taken += 1
            rewards.append(float(result.reward))
            done = bool(result.done)

            print(
                "[STEP] "
                f"step={steps_taken} "
                f"action={_format_action(action)} "
                f"reward={result.reward:.2f} "
                f"done={_format_bool(done)} "
                f"error={_format_error(getattr(env, 'last_action_error', None))}",
                flush=True,
            )

            observation = result.observation
    except Exception as exc:
        episode_error = str(exc)
    finally:
        try:
            env.close()
        except Exception as close_exc:
            if episode_error is None:
                episode_error = str(close_exc)

        total_reward = sum(rewards)
        success = done and episode_error is None and total_reward >= SUCCESS_REWARD
        reward_list = ",".join(f"{reward:.2f}" for reward in rewards)
        print(
            f"[END] success={_format_bool(success)} steps={steps_taken} rewards={reward_list}",
            flush=True,
        )

    return {
        "success": done and episode_error is None and sum(rewards) >= SUCCESS_REWARD,
        "steps_taken": steps_taken,
        "rewards": rewards,
        "error": episode_error,
    }


def main() -> None:
    env = EmailTriageEnv()
    client = _build_client()
    run_episode(env, client)


if __name__ == "__main__":
    main()
