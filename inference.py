from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from env.environment import EmailTriageEnv, ACTION_SEQUENCE
from env.models import Action, Observation

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "openenv/email-triage:latest")

MAX_STEPS = 10
MAX_TOTAL_REWARD = 1.0
SUCCESS_THRESHOLD = 0.8

SYSTEM_PROMPT = """\
You are an email triage agent. You process customer emails in a strict 3-step pipeline.

## Pipeline

Step 1 - classify_email: Classify the email into exactly one category.
  Categories: complaint, query, request
  Rules:
    - "refund", "unacceptable", "damaged" -> complaint
    - "status", "how", "where", "when", "password" -> query
    - everything else -> request

Step 2 - set_priority: Assign a priority based on the category.
  Rules:
    - complaint -> high
    - query    -> medium
    - request  -> low

Step 3 - generate_reply: Generate the reply based on the priority.
  Rules:
    - high   -> "We are processing your refund."
    - medium -> "Your query has been noted and we will respond shortly."
    - low    -> "We have received your request and will follow up."

## Response Format

Respond with ONLY a JSON object (no markdown, no explanation):
{"action_type": "<action>", "content": "<value>"}

Where <action> is the next required action in the pipeline.\
"""


def log_start(email: str, model: str, image: str) -> None:
    logger.info("=" * 60)
    logger.info("EPISODE START")
    logger.info("  model : %s", model)
    logger.info("  image : %s", image)
    logger.info("  email : %.80s...", email.replace("\n", " ") if len(email) > 80 else email)
    logger.info("=" * 60)


def log_step(
    step: int,
    action_type: str,
    content: str,
    reward: float,
    done: bool,
    info: dict[str, Any],
) -> None:
    match_str = "MATCH" if info.get("match") else "MISS"
    if "error" in info:
        match_str = "ERROR"
    logger.info(
        "  step %d | %-16s | %-40s | reward=%+.2f | %s | done=%s",
        step, action_type, content[:40], reward, match_str, done,
    )


def log_end(
    steps_taken: int,
    total_reward: float,
    score: float,
    success: bool,
    elapsed: float,
) -> None:
    logger.info("=" * 60)
    logger.info("EPISODE END")
    logger.info("  steps   : %d", steps_taken)
    logger.info("  reward  : %.2f", total_reward)
    logger.info("  score   : %.4f", score)
    logger.info("  success : %s", success)
    logger.info("  elapsed : %.2fs", elapsed)
    logger.info("=" * 60)


def _build_user_message(observation: Observation, step_index: int) -> str:
    parts = [f"Email: {observation.email}"]
    if observation.category:
        parts.append(f"Assigned category: {observation.category}")
    if observation.priority:
        parts.append(f"Assigned priority: {observation.priority}")

    next_action = ACTION_SEQUENCE[step_index] if step_index < len(ACTION_SEQUENCE) else None
    if next_action:
        parts.append(f"\nNext required action: {next_action}")

    return "\n".join(parts)


def _parse_action(raw: str) -> Action | None:
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return Action(
            action_type=data["action_type"],
            content=data.get("content", ""),
        )
    except (json.JSONDecodeError, KeyError, Exception):
        return None


def _fallback_action(observation: Observation, step_index: int) -> Action:
    email_lower = observation.email.lower()

    if step_index == 0:
        if any(kw in email_lower for kw in ("refund", "damaged", "unacceptable")):
            category = "complaint"
        elif any(kw in email_lower for kw in ("how", "status", "where", "when", "password")):
            category = "query"
        else:
            category = "request"
        return Action(action_type="classify_email", content=category)

    if step_index == 1:
        cat = (observation.category or "").lower()
        priority_map = {"complaint": "high", "query": "medium"}
        return Action(action_type="set_priority", content=priority_map.get(cat, "low"))

    pri = (observation.priority or "").lower()
    reply_map = {
        "high": "We are processing your refund.",
        "medium": "Your query has been noted and we will respond shortly.",
    }
    return Action(
        action_type="generate_reply",
        content=reply_map.get(pri, "We have received your request and will follow up."),
    )


def generate_action(
    client: OpenAI,
    observation: Observation,
    step_index: int,
    history: list[dict[str, str]],
) -> Action:
    expected_action = ACTION_SEQUENCE[step_index] if step_index < len(ACTION_SEQUENCE) else None
    user_msg = _build_user_message(observation, step_index)

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content or ""
        action = _parse_action(raw)
    except Exception:
        action = None
        raw = ""

    if action is None or (expected_action and action.action_type != expected_action):
        action = _fallback_action(observation, step_index)
        logger.warning("  fallback | expected=%s | llm_gave=%s", expected_action, raw[:60])

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": json.dumps({"action_type": action.action_type, "content": action.content})})

    return action


def run_episode(
    env: EmailTriageEnv,
    client: OpenAI,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    observation = env.reset()
    log_start(observation.email, MODEL_NAME, IMAGE_NAME)

    history: list[dict[str, str]] = []
    rewards: list[float] = []
    actions_taken: list[dict[str, str]] = []
    steps_taken = 0
    done = False

    while not done and steps_taken < MAX_STEPS:
        step_index = env._step_index

        action = generate_action(client, observation, step_index, history)

        result = env.step(action)
        steps_taken += 1

        rewards.append(result.reward)
        actions_taken.append({"action_type": action.action_type, "content": action.content or ""})

        log_step(
            step=steps_taken,
            action_type=action.action_type,
            content=action.content or "",
            reward=result.reward,
            done=result.done,
            info=result.info,
        )

        observation = result.observation
        done = result.done

    total_reward = sum(rewards)
    score = max(0.0, min(1.0, total_reward / MAX_TOTAL_REWARD))
    success = score >= SUCCESS_THRESHOLD
    elapsed = time.perf_counter() - t0

    log_end(steps_taken, total_reward, score, success, elapsed)

    return {
        "score": round(score, 4),
        "success": success,
        "total_reward": round(total_reward, 2),
        "steps_taken": steps_taken,
        "rewards": rewards,
        "actions": actions_taken,
        "final_state": env.state(),
    }


def main() -> None:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

    env = EmailTriageEnv()
    result = run_episode(env, client)

    print("\n--- RESULT ---")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
