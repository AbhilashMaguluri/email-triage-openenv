# OpenEnv Email Triage Environment

An AI agent benchmarking environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. It simulates a real-world customer email triage workflow where an AI agent must **classify** incoming emails, **assign** priority levels, and **generate** appropriate replies -- then evaluates the agent's performance with deterministic grading across three difficulty tiers.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Reward Mechanics](#reward-mechanics)
  - [Business Rules](#business-rules)
  - [Sample Emails](#sample-emails)
  - [Episode Lifecycle](#episode-lifecycle)
- [Data Models](#data-models)
  - [Observation](#observation)
  - [Action](#action)
  - [StepResult](#stepresult)
- [Tasks & Grading System](#tasks--grading-system)
  - [Task Difficulty Tiers](#task-difficulty-tiers)
  - [Grader Functions](#grader-functions)
  - [Score Breakdown](#score-breakdown)
  - [Task Registry & API](#task-registry--api)
  - [Oracle Runs](#oracle-runs)
- [Inference Engine](#inference-engine)
  - [Configuration](#configuration)
  - [LLM Integration](#llm-integration)
  - [Episode Flow](#episode-flow)
  - [Logging Format](#logging-format)
  - [Scoring & Success Criteria](#scoring--success-criteria)
- [OpenEnv Manifest](#openenv-manifest)
- [Testing](#testing)
  - [Environment Tests](#environment-tests)
  - [Task & Grader Tests](#task--grader-tests)
  - [Inference Validation](#inference-validation)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

This environment provides a standardized, reproducible benchmark for evaluating how well AI agents handle the common enterprise task of email triage. The pipeline mirrors real customer support workflows:

```
Incoming Email --> Classify Category --> Assign Priority --> Generate Reply
```

The environment enforces a strict 3-step sequential pipeline. At each step, the agent must submit the correct action type in the correct order. The environment then compares the agent's output against deterministic business rules and awards or penalizes accordingly.

**Key features:**

- **Gymnasium-style API** -- familiar `reset()` / `step()` / `state()` interface
- **Pydantic-validated models** -- type-safe `Action`, `Observation`, and `StepResult` with runtime validation
- **3-tier task system** -- Easy, Medium, and Hard difficulty levels with independent deterministic graders
- **LLM inference engine** -- plug-and-play async inference with any OpenAI-compatible API
- **OpenEnv-compatible manifest** -- ready for containerized deployment via `openenv build` and `openenv push`

---

## Project Structure

```
email-triage-env/
|
|-- openenv.yaml              # OpenEnv manifest (metadata, tasks, build config)
|
|-- env/                      # Core environment package
|   |-- __init__.py           # Public exports: EmailTriageEnv, Action, Observation, StepResult
|   |-- models.py             # Pydantic data models (Observation, Action, StepResult)
|   |-- environment.py        # EmailTriageEnv class, business rules, reward logic
|
|-- tasks/                    # Evaluation tasks & grading package
|   |-- __init__.py           # Public exports: Task, TaskResult, run_all_tasks, etc.
|   |-- tasks.py              # Task definitions, grader functions, registry, runner
|
|-- inference.py              # LLM-driven inference engine (async, OpenAI-compatible)
|
|-- test_env.py               # Environment unit tests
|-- test_tasks.py             # Task & grader unit tests
|-- validate_inference.py     # Inference module smoke tests
```

---

## Architecture

The system is composed of four layers, each decoupled and independently testable:

```
+---------------------------------------------------------------+
|                        INFERENCE LAYER                        |
|  inference.py -- LLM client, prompt engineering, episode loop |
+---------------------------------------------------------------+
                              |
                     Action (JSON from LLM)
                              |
                              v
+---------------------------------------------------------------+
|                        TASK LAYER                             |
|  tasks/tasks.py -- Task definitions, Graders, Registry        |
+---------------------------------------------------------------+
                              |
                    execute() -> grade()
                              |
                              v
+---------------------------------------------------------------+
|                      ENVIRONMENT LAYER                        |
|  env/environment.py -- EmailTriageEnv (reset/step/state)      |
+---------------------------------------------------------------+
                              |
                  Pydantic validation
                              |
                              v
+---------------------------------------------------------------+
|                        MODEL LAYER                            |
|  env/models.py -- Observation, Action, StepResult             |
+---------------------------------------------------------------+
```

**Data flow for a single episode:**

1. `inference.py` calls `env.reset()` and receives an `Observation` containing the email body
2. The email is sent to the LLM with a system prompt describing the pipeline rules
3. The LLM returns a JSON action, which is parsed into an `Action` Pydantic model
4. `env.step(action)` validates the action, computes reward, updates observation, returns `StepResult`
5. Steps 2-4 repeat until `done=True` or `MAX_STEPS` is reached
6. Final score is computed: `score = clamp(sum(rewards) / MAX_TOTAL_REWARD, 0, 1)`

---

## Installation

### Prerequisites

- Python 3.10+
- pip or uv

### Install Dependencies

```bash
pip install pydantic openai
```

Or with uv:

```bash
uv pip install pydantic openai
```

### Verify Installation

```bash
python test_env.py
python test_tasks.py
python validate_inference.py
```

All three scripts should print their respective "All tests passed!" message.

---

## Quick Start

### 1. Run the environment manually

```python
from env import EmailTriageEnv, Action

env = EmailTriageEnv()
obs = env.reset()
print(f"Email: {obs.email}")

# Step 1: Classify
result = env.step(Action(action_type="classify_email", content="complaint"))
print(f"Reward: {result.reward}, Done: {result.done}")

# Step 2: Set priority
result = env.step(Action(action_type="set_priority", content="high"))
print(f"Reward: {result.reward}, Done: {result.done}")

# Step 3: Generate reply
result = env.step(Action(action_type="generate_reply", content="We are processing your refund."))
print(f"Reward: {result.reward}, Done: {result.done}")

# Check final state
state = env.state()
print(f"Total reward: {state['total_reward']}")
```

### 2. Run all tasks with oracle answers

```python
from tasks import run_all_tasks

results = run_all_tasks()
for level, result in results.items():
    print(f"{level}: score={result.score}, passed={result.passed}")
```

### 3. Run LLM-powered inference

```bash
# Set environment variables
set API_BASE_URL=https://api.openai.com/v1
set OPENAI_API_KEY=sk-your-key-here
set MODEL_NAME=gpt-4o-mini

# Run inference
python inference.py
```

---

## Environment Details

### Observation Space

The observation is a Pydantic model that progressively accumulates data as the agent completes each pipeline step:

| Field | Type | Initial Value | Populated By |
|-------|------|---------------|--------------|
| `email` | `str` | Raw email text | `reset()` |
| `category` | `str \| None` | `None` | `classify_email` action |
| `priority` | `str \| None` | `None` | `set_priority` action |
| `reply` | `str \| None` | `None` | `generate_reply` action |

### Action Space

Actions are discrete and must follow a strict sequential order:

| Step | Action Type | Content | Description |
|------|-------------|---------|-------------|
| 1 | `classify_email` | `complaint` \| `query` \| `request` | Classify email category |
| 2 | `set_priority` | `high` \| `medium` \| `low` | Assign priority level |
| 3 | `generate_reply` | String | Generate reply text |

The `action_type` field is validated against the regex pattern `^(classify_email|set_priority|generate_reply)$` at the Pydantic model level. Submitting an action out of sequence results in a `-0.2` penalty without advancing the pipeline.

### Reward Mechanics

| Scenario | Reward |
|----------|--------|
| Correct classification | `+0.4` |
| Correct priority | `+0.3` |
| Correct reply (exact match) | `+0.3` |
| Wrong classification | `-0.2` |
| Wrong priority | `-0.2` |
| Wrong reply | `-0.2` |
| Wrong action order | `-0.2` |

**Maximum total reward for a perfect episode: `+1.0`** (0.4 + 0.3 + 0.3)

**Minimum possible reward: `-0.6`** (all three wrong: -0.2 x 3)

### Business Rules

The environment uses deterministic keyword-based rules to derive expected outputs from the email body. These rules are the ground truth that graders evaluate against.

#### Classification Rules

| Condition | Category |
|-----------|----------|
| Email contains "refund", "unacceptable", or "damaged" | `complaint` |
| Email contains "status", "how", "where", "when", or "password" | `query` |
| Everything else | `request` |

Keywords are matched case-insensitively against the full email body.

#### Priority Rules

| Category | Priority |
|----------|----------|
| `complaint` | `high` |
| `query` | `medium` |
| `request` | `low` |

Priority is derived directly from the category. It is a deterministic 1:1 mapping.

#### Reply Rules

| Priority | Expected Reply |
|----------|----------------|
| `high` | `"We are processing your refund."` |
| `medium` | `"Your query has been noted and we will respond shortly."` |
| `low` | `"We have received your request and will follow up."` |

Replies are evaluated by **exact string match**. Agents must produce the reply character-for-character.

### Sample Emails

The environment ships with 6 built-in sample emails covering all three categories:

| # | Email Body | Category | Priority |
|---|-----------|----------|----------|
| 1 | "I want a refund for my last order. The product was damaged." | complaint | high |
| 2 | "Can you tell me the status of my order #12345?" | query | medium |
| 3 | "I'd like to request a demo of your enterprise plan." | request | low |
| 4 | "This is unacceptable! I demand a refund immediately." | complaint | high |
| 5 | "How do I reset my account password?" | query | medium |
| 6 | "Please add me to your newsletter mailing list." | request | low |

You can supply custom emails to the constructor:

```python
custom_emails = [
    {
        "body": "Your service is terrible, I want my money back!",
        "expected_category": "complaint",
        "expected_priority": "high",
        "expected_reply": "We are processing your refund.",
    }
]
env = EmailTriageEnv(emails=custom_emails)
```

### Episode Lifecycle

```
          +------------------+
          |   env.reset()    |  <-- Selects random email, derives expected outputs
          +--------+---------+
                   |
                   v
          +------------------+
  Step 1  | classify_email   |  <-- Agent provides category
          +--------+---------+
                   |
                   v
          +------------------+
  Step 2  | set_priority     |  <-- Agent provides priority
          +--------+---------+
                   |
                   v
          +------------------+
  Step 3  | generate_reply   |  <-- Agent provides reply, episode ends
          +--------+---------+
                   |
                   v
          +------------------+
          |   done = True    |  <-- Final state available via env.state()
          +------------------+
```

If the agent submits the wrong action type at any step, the environment returns a `-0.2` penalty but does **not** advance the step counter. The agent can retry (up to `MAX_STEPS`).

---

## Data Models

All models are defined in `env/models.py` using Pydantic `BaseModel` for runtime validation and serialization.

### Observation

```python
class Observation(BaseModel):
    email: str                       # Raw email body (always present)
    category: Optional[str] = None   # Set after classify_email
    priority: Optional[str] = None   # Set after set_priority
    reply: Optional[str] = None      # Set after generate_reply
```

### Action

```python
class Action(BaseModel):
    action_type: str = Field(
        ...,
        pattern=r"^(classify_email|set_priority|generate_reply)$",
    )
    content: Optional[str] = None
```

The `action_type` is validated against the regex at construction time. Invalid action types raise a Pydantic `ValidationError`.

### StepResult

```python
class StepResult(BaseModel):
    observation: Observation          # Updated observation after the step
    reward: float                     # Reward for this step (-0.2 to +0.4)
    done: bool                       # True when episode is complete
    info: dict[str, Any]             # Diagnostics: step number, match status, errors
```

The `info` dict contains:

| Key | Type | Present When | Description |
|-----|------|-------------|-------------|
| `step` | `int` | Always | Current step number (1-indexed) |
| `match` | `bool` | Correct action type | Whether content matched expected value |
| `expected` | `str` | Wrong content | The expected value (for debugging) |
| `error` | `str` | Wrong action order | Error message explaining the mismatch |
| `total_reward` | `float` | Episode end | Cumulative reward for the episode |

---

## Tasks & Grading System

The task system provides a structured way to evaluate agents at different capability levels. Tasks are defined in `tasks/tasks.py`.

### Task Difficulty Tiers

| Difficulty | Task Name | Required Actions | What it Tests |
|------------|-----------|------------------:|---------------|
| **Easy** | Email Classification | `classify_email` | Can the agent understand email intent? |
| **Medium** | Email Classification + Priority | `classify_email`, `set_priority` | Can the agent apply business rules? |
| **Hard** | Full Email Triage Pipeline | `classify_email`, `set_priority`, `generate_reply` | Can the agent complete the full workflow? |

### Grader Functions

Each task has a dedicated, deterministic, pure grader function that takes the environment state dict and returns a score in `[0.0, 1.0]`.

#### Easy Grader (`_grade_easy`)

```
return 1.0 if category correct else 0.0
```

Binary pass/fail on classification accuracy.

#### Medium Grader (`_grade_medium`)

```
score  = 0.5 if category correct
score += 0.5 if priority correct
return score
```

Equal weight between classification and prioritization.

#### Hard Grader (`_grade_hard`)

```
score  = 0.4 if category correct
score += 0.3 if priority correct
score += 0.3 if reply correct (exact match)
return score
```

Classification is weighted slightly higher because downstream steps depend on it.

### Score Breakdown

Every `TaskResult` includes a detailed `score_breakdown` in its `details` dict. This provides per-criterion diagnostics:

```python
result = run_task("hard")
for criterion, info in result.details["score_breakdown"].items():
    print(f"{criterion}:")
    print(f"  weight:   {info['weight']}")
    print(f"  matched:  {info['matched']}")
    print(f"  earned:   {info['earned']}")
    print(f"  actual:   {info['actual']}")
    print(f"  expected: {info['expected']}")
```

Example output:

```
category:
  weight:   0.4
  matched:  True
  earned:   0.4
  actual:   complaint
  expected: complaint
priority:
  weight:   0.3
  matched:  True
  earned:   0.3
  actual:   high
  expected: high
reply:
  weight:   0.3
  matched:  True
  earned:   0.3
  actual:   We are processing your refund.
  expected: We are processing your refund.
```

### Task Registry & API

Tasks are organized in a registry indexed by `TaskDifficulty`:

```python
from tasks import get_task, run_task, run_all_tasks, TaskDifficulty

# Get a task object
task = get_task("hard")                    # or get_task(TaskDifficulty.HARD)

# Run a single task
result = run_task("easy")                  # Returns TaskResult

# Run all tasks
scores = run_all_tasks()                   # Returns dict[str, TaskResult]
scores["easy"].score                       # 1.0
scores["medium"].passed                    # True
scores["hard"].details["score_breakdown"]  # per-criterion details
```

### Oracle Runs

Tasks support **oracle / gold-standard** runs via the `__FROM_EXPECTED__` sentinel. When an action's `content` is set to `"__FROM_EXPECTED__"`, the task runner automatically substitutes the correct expected value at runtime. This allows you to verify that the environment + grader pipeline is functioning correctly:

```python
from tasks import run_all_tasks

# All pre-built tasks use __FROM_EXPECTED__ -- they always score 1.0
results = run_all_tasks()
assert all(r.passed for r in results.values())
```

To test with deliberately wrong answers, create a custom task:

```python
from tasks.tasks import Task, _grade_easy
from tasks import TaskDifficulty
from env.models import Action

wrong_task = Task(
    name="Intentionally Wrong",
    difficulty=TaskDifficulty.EASY,
    actions=[Action(action_type="classify_email", content="query")],  # Wrong!
    grader=_grade_easy,
)
result = wrong_task.execute(env)
print(result.score)  # 0.0
```

---

## Inference Engine

The `inference.py` module provides a complete, production-ready inference loop that connects an LLM to the environment via the OpenAI chat completions API.

### Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API endpoint |
| `OPENAI_API_KEY` | `""` | API authentication key |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `IMAGE_NAME` | `openenv/email-triage:latest` | Docker image name (for logging) |

And in-code constants:

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_STEPS` | `10` | Maximum steps per episode before forced termination |
| `MAX_TOTAL_REWARD` | `1.0` | Normalizer for score calculation |
| `SUCCESS_THRESHOLD` | `0.8` | Minimum score for `success = True` |

### LLM Integration

The inference engine uses a carefully crafted system prompt that encodes the complete business rules:

```
System Prompt Structure:
1. Role definition ("You are an email triage agent")
2. Pipeline specification (3 steps with exact rules)
3. Classification rules (keyword -> category mapping)
4. Priority rules (category -> priority mapping)
5. Reply rules (priority -> exact reply text)
6. Response format (strict JSON: {"action_type": "...", "content": "..."})
```

At each step, the engine builds a context-aware user message containing:

- The original email text
- Previously assigned category (if any)
- Previously assigned priority (if any)
- The next required action type

**Conversation history** is maintained across steps within an episode, giving the LLM full context of its previous decisions.

The `_parse_action()` function handles LLM output robustly:

- Strips whitespace
- Removes markdown code fences (` ```json ... ``` `)
- Parses JSON into an `Action` model

### Episode Flow

```python
async def main():
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
    env = EmailTriageEnv()
    result = await run_episode(env, client)
    print(json.dumps(result, indent=2))
```

The `run_episode()` function returns a comprehensive result dict:

```json
{
  "score": 1.0,
  "success": true,
  "total_reward": 1.0,
  "steps_taken": 3,
  "rewards": [0.4, 0.3, 0.3],
  "actions": [
    {"action_type": "classify_email", "content": "complaint"},
    {"action_type": "set_priority", "content": "high"},
    {"action_type": "generate_reply", "content": "We are processing your refund."}
  ],
  "final_state": { ... }
}
```

### Logging Format

The inference engine uses structured logging with three dedicated functions:

#### `log_start(email, model, image)`

```
16:30:01 | INFO    | ============================================================
16:30:01 | INFO    | EPISODE START
16:30:01 | INFO    |   model : gpt-4o-mini
16:30:01 | INFO    |   image : openenv/email-triage:latest
16:30:01 | INFO    |   email : I want a refund for my last order. The product was damaged.
16:30:01 | INFO    | ============================================================
```

#### `log_step(step, action_type, content, reward, done, info)`

```
16:30:02 | INFO    |   step 1 | classify_email   | complaint          | reward=+0.40 | MATCH | done=False
16:30:03 | INFO    |   step 2 | set_priority     | high               | reward=+0.30 | MATCH | done=False
16:30:04 | INFO    |   step 3 | generate_reply   | We are processing  | reward=+0.30 | MATCH | done=True
```

#### `log_end(steps_taken, total_reward, score, success, elapsed)`

```
16:30:04 | INFO    | ============================================================
16:30:04 | INFO    | EPISODE END
16:30:04 | INFO    |   steps   : 3
16:30:04 | INFO    |   reward  : 1.00
16:30:04 | INFO    |   score   : 1.0000
16:30:04 | INFO    |   success : True
16:30:04 | INFO    |   elapsed : 2.34s
16:30:04 | INFO    | ============================================================
```

### Scoring & Success Criteria

```python
total_reward = sum(rewards)                              # Sum of per-step rewards
score = max(0.0, min(1.0, total_reward / MAX_TOTAL_REWARD))  # Clamp to [0, 1]
success = score >= SUCCESS_THRESHOLD                     # Default threshold: 0.8
```

| Scenario | total_reward | score | success |
|----------|-------------|-------|---------|
| Perfect run | 1.0 | 1.0 | True |
| 2/3 correct, 1 wrong | 0.5 | 0.5 | False |
| All wrong | -0.6 | 0.0 | False |
| 1 correct + 2 wrong | 0.0 | 0.0 | False |

---

## OpenEnv Manifest

The `openenv.yaml` file is the project manifest that makes this environment compatible with the OpenEnv ecosystem:

```yaml
name: email-triage-env
version: "1.0"
description: >
  AI Email Triage & Response Environment that classifies emails,
  assigns priority, and generates replies.

metadata:
  author: Abhilash Maguluri
  license: MIT
  tags: [email, triage, classification, nlp]

entry_point: env.environment:EmailTriageEnv

tasks:
  - id: easy
    grader: tasks.tasks:_grade_easy
    scoring: { category: 1.0 }

  - id: medium
    grader: tasks.tasks:_grade_medium
    scoring: { category: 0.5, priority: 0.5 }

  - id: hard
    grader: tasks.tasks:_grade_hard
    scoring: { category: 0.4, priority: 0.3, reply: 0.3 }

build:
  dockerfile: Dockerfile
  requirements: requirements.txt

deploy:
  port: 8000
  healthcheck: /health
```

### OpenEnv CLI Workflow

```bash
# Initialize (if starting fresh)
openenv init email-triage-env

# Validate manifest
openenv validate

# Build Docker image
openenv build

# Deploy to Hugging Face Spaces
openenv push
```

---

## Testing

### Environment Tests

```bash
python test_env.py
```

Covers:

| Test | What it Verifies |
|------|-----------------|
| Perfect run | All 3 steps with correct answers yield reward=1.0 |
| Wrong action order | Submitting `set_priority` first yields -0.2 penalty |
| Wrong classification | Submitting wrong category yields -0.2 penalty |

### Task & Grader Tests

```bash
python test_tasks.py
```

Covers:

| Test | What it Verifies |
|------|-----------------|
| Oracle run (all levels) | All 3 tasks score 1.0 with gold-standard answers |
| Wrong classification | Easy task correctly scores 0.0 for wrong category |
| Grader isolation (fully correct) | All graders return 1.0 |
| Grader isolation (category only) | easy=1.0, medium=0.5, hard=0.4 |
| Grader isolation (all wrong) | All graders return 0.0 |
| Grader isolation (priority only) | easy=0.0, medium=0.5, hard=0.3 |
| Grader isolation (category+reply) | hard=0.7 |
| run_task convenience | All difficulty levels pass with oracle answers |

### Inference Validation

```bash
python validate_inference.py
```

Covers:

| Test | What it Verifies |
|------|-----------------|
| `_parse_action` (clean JSON) | Parses `{"action_type": "...", "content": "..."}` |
| `_parse_action` (markdown fenced) | Strips ` ```json ``` ` wrappers |
| `_build_user_message` (step 0) | Includes email and next action `classify_email` |
| `_build_user_message` (step 1) | Includes assigned category, next action `set_priority` |
| `_build_user_message` (step 2) | Includes assigned priority, next action `generate_reply` |
| Logging functions | `log_start`, `log_step`, `log_end` execute without error |
| Constants | `MAX_STEPS=10`, `MAX_TOTAL_REWARD=1.0`, `SUCCESS_THRESHOLD=0.8` |
| Score clamping | Values >1 clamp to 1.0, values <0 clamp to 0.0 |

---

## API Reference

### `env.EmailTriageEnv`

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `(emails: list[dict] \| None = None)` | `None` | Initialize with custom or default emails |
| `reset` | `()` | `Observation` | Start a new episode with a random email |
| `step` | `(action: Action)` | `StepResult` | Execute one action and advance the pipeline |
| `state` | `()` | `dict[str, Any]` | Get full environment state (observation, expected, reward) |

### `tasks` Module

| Function | Signature | Returns | Description |
|----------|-----------|---------|-------------|
| `get_task` | `(difficulty: str \| TaskDifficulty)` | `Task` | Retrieve task definition by difficulty |
| `run_task` | `(difficulty, env=None)` | `TaskResult` | Execute a single task and return graded result |
| `run_all_tasks` | `(env=None)` | `dict[str, TaskResult]` | Execute all tasks, returns `{"easy": ..., "medium": ..., "hard": ...}` |

### FastAPI Environment Wrapper (`app.py`)

The environment is wrapped in a FastAPI service for remote access.

| Endpoint | Method | Payload | Returns | Description |
|----------|--------|---------|---------|-------------|
| `/` | GET | `None` | `{"message": "..."}` | Root endpoint to confirm service is running |
| `/health`| GET | `None` | `{"status": "ok"}` | Healthcheck |
| `/reset` | GET | `None` | `Observation` (JSON) | Initialize or reset the environment episode |
| `/step` | POST | `{"action_type": str, "content": str}` | JSON containing `observation`, `reward`, `done`, `info` | Take a pipeline action within the environment |

**Local Startup:**

```bash
# Install dependencies
pip install -r requirements.txt

# Start Server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### `TaskResult` Fields

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | Human-readable name |
| `difficulty` | `TaskDifficulty` | easy / medium / hard |
| `score` | `float` | Grader output in [0.0, 1.0] |
| `max_score` | `float` | Always 1.0 |
| `passed` | `bool` (property) | `True` when `score >= max_score` |
| `details` | `dict` | Contains `observation`, `expected`, `total_reward`, `score_breakdown` |

### `inference` Module

| Function | Signature | Returns | Description |
|----------|-----------|---------|-------------|
| `run_episode` | `async (env, client)` | `dict[str, Any]` | Run full episode with LLM agent |
| `generate_action` | `async (client, obs, step_idx, history)` | `Action` | Get next action from LLM |
| `main` | `async ()` | `None` | Entry point: creates client + env, runs episode |

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Used By |
|----------|----------|---------|---------|
| `API_BASE_URL` | No | `https://api.openai.com/v1` | `inference.py` |
| `OPENAI_API_KEY` | Yes (for inference) | `""` | `inference.py` |
| `MODEL_NAME` | No | `gpt-4o-mini` | `inference.py` |
| `IMAGE_NAME` | No | `openenv/email-triage:latest` | `inference.py` (logging only) |

### In-Code Constants

| Constant | File | Value | Description |
|----------|------|-------|-------------|
| `MAX_STEPS` | `inference.py` | `10` | Max steps before forced episode end |
| `MAX_TOTAL_REWARD` | `inference.py` | `1.0` | Score normalizer |
| `SUCCESS_THRESHOLD` | `inference.py` | `0.8` | Minimum score to be considered successful |
| `ACTION_SEQUENCE` | `env/environment.py` | `["classify_email", "set_priority", "generate_reply"]` | Required action order |

---

## Examples

### Example 1: Evaluate a custom agent function

```python
from env import EmailTriageEnv, Action

def my_agent(observation):
    """A simple rule-based agent."""
    email = observation.email.lower()

    if "refund" in email or "damaged" in email:
        category = "complaint"
    elif "status" in email or "how" in email:
        category = "query"
    else:
        category = "request"

    priority = {"complaint": "high", "query": "medium", "request": "low"}[category]

    replies = {
        "high": "We are processing your refund.",
        "medium": "Your query has been noted and we will respond shortly.",
        "low": "We have received your request and will follow up.",
    }
    reply = replies[priority]

    return [
        Action(action_type="classify_email", content=category),
        Action(action_type="set_priority", content=priority),
        Action(action_type="generate_reply", content=reply),
    ]

env = EmailTriageEnv()
obs = env.reset()
actions = my_agent(obs)

for action in actions:
    result = env.step(action)
    print(f"{action.action_type}: reward={result.reward}")

print(f"Total: {env.state()['total_reward']}")
```

### Example 2: Batch evaluation across all emails

```python
from env import EmailTriageEnv
from tasks import run_all_tasks

env = EmailTriageEnv()
all_scores = {"easy": [], "medium": [], "hard": []}

for _ in range(100):
    results = run_all_tasks(env)
    for level, result in results.items():
        all_scores[level].append(result.score)

for level, scores in all_scores.items():
    avg = sum(scores) / len(scores)
    print(f"{level}: avg_score={avg:.2f}")
```

### Example 3: Use with a different LLM provider

```bash
# Use with Ollama
set API_BASE_URL=http://localhost:11434/v1
set OPENAI_API_KEY=ollama
set MODEL_NAME=llama3

python inference.py
```

```bash
# Use with Azure OpenAI
set API_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
set OPENAI_API_KEY=your-azure-key
set MODEL_NAME=gpt-4o

python inference.py
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run all tests: `python test_env.py && python test_tasks.py && python validate_inference.py`
5. Submit a pull request

### Adding a New Task

1. Define a grader function in `tasks/tasks.py`:
   ```python
   def _grade_custom(state: dict[str, Any]) -> float:
       # Your scoring logic here
       return score
   ```

2. Create a `Task` instance:
   ```python
   TASK_CUSTOM = Task(
       name="Custom Task",
       difficulty=TaskDifficulty.HARD,
       actions=[...],
       grader=_grade_custom,
   )
   ```

3. Register it in `TASK_REGISTRY`

### Adding New Emails

Add entries to `SAMPLE_EMAILS` in `env/environment.py` following the existing format. The `expected_*` fields in the dict are for reference only -- the environment derives expected values dynamically via `_derive_expected()`.

---

## License

MIT License

---

## Author

**Abhilash Maguluri**

Built with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.
