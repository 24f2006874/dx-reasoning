---
title: Dx Reasoning Environment Server
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - medical
  - diagnosis
  - healthcare
---

# Dx Reasoning Environment

A comprehensive clinical diagnostic reasoning environment for training AI agents to diagnose like doctors. Agents must gather patient information through history, questions, physical exams, and tests before making a final diagnosis.

## Features

### 🏥 Comprehensive Disease Knowledge Base
- **11+ diseases** in the case library
- **Submission task set**: easy, medium, hard
- **Extended internal cases** include expert-level scenarios for local experimentation
- **Realistic patient profiles** with demographics, vital signs, and medical history
- **Detailed symptom profiles** with probability-based occurrence
- **Differential diagnoses** and teaching points for each disease

### 🧠 Intelligent Response Engine
- **Context-aware history responses** based on disease and patient profile
- **Dynamic physical exam findings** that vary by disease
- **Realistic diagnostic test results** with proper medical values
- **Progressive information disclosure** based on questioning strategy

### 📊 Enhanced Reward System
- **Base rewards** for information gathering (0.25-0.35)
- **High reward** for correct diagnosis (0.85)
- **Efficiency bonus** (+0.10) for steps 7+ to encourage thorough workups
- **Comprehensive history bonus** for requesting full history

### 🎯 Submission Difficulty Levels
| Level | Diseases | Hints | Description |
|-------|----------|-------|-------------|
| Easy | 4 | 2 | Classic presentations (strep throat, appendicitis, pneumonia, DKA) |
| Medium | 4 | 1 | Atypical presentations (atypical MI, PE, meningitis, ectopic pregnancy) |
| Hard | 1 | 0 | Complex multi-system (myasthenia gravis) |

## Quick Start

The simplest way to use the Dx Reasoning environment is through the `DxReasoningEnv` async client:

```python
import asyncio

from client import DxReasoningEnv
from models import ActionType, DxAction


async def run_episode() -> None:
    async with DxReasoningEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task="easy")
        obs = result.observation
        print(f"Patient: {obs.patient_context}")
        print(f"Notes: {obs.clinical_notes}")

        for action in [
            DxAction(action_type=ActionType.REQUEST_HISTORY, content="full history"),
            DxAction(action_type=ActionType.ASK_QUESTION, content="symptom details"),
            DxAction(action_type=ActionType.DIAGNOSE, content="strep throat"),
        ]:
            step_result = await env.step(action)
            print(f"Action: {action.action_type.value}")
            print(f"  -> Notes: {step_result.observation.clinical_notes}")
            print(f"  -> Reward: {step_result.reward}")
            print(f"  -> Done: {step_result.done}")
            if step_result.done:
                break


asyncio.run(run_episode())
```

The client uses a persistent session connection to a running environment server and returns `StepResult` objects for `reset` and `step`.

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t dx_reasoning-env:latest -f Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action Types

| Action | Description | Reward |
|--------|-------------|--------|
| `request_history` | Request patient medical history | 0.30 |
| `ask_question` | Ask specific questions about symptoms | 0.30 |
| `physical_exam` | Perform physical examination | 0.25 |
| `order_test` | Order diagnostic tests (lab, imaging) | 0.25 |
| `diagnose` | Submit final diagnosis | 0.85 (correct) / 0.40 (incorrect) |

### Observation Fields

**DxObservation** contains:
- `done` (bool) - Episode complete
- `reward` (float) - Step reward (0.0-1.0)
- `patient_context` (str) - Patient information
- `test_results` (dict) - Lab/imaging results
- `exam_findings` (list) - Physical exam findings
- `history_details` (list) - Patient history info
- `clinical_notes` (str) - Doctor's notes
- `hints_remaining` (int) - Hints available
- `warning` (str) - Warning messages

### Difficulty Levels

| Level | Description | Cases | Hints |
|-------|-------------|-------|-------|
| `easy` | Textbook presentations | strep_throat, appendicitis | 2 |
| `medium` | Overlapping symptoms | atypical_mi, pulmonary_embolism | 1 |
| `hard` | Complex multi-system | myasthenia_gravis | 0 |

### Reward System

- Base step reward: 0.15
- Information gathering (history/questions): 0.30
- Actions (exam/tests): 0.25
- Correct diagnosis: 0.85
- Incorrect diagnosis: 0.40
- Late steps (≥7): +0.10 bonus

## Advanced Usage

### Connecting to an Existing Server

If you already have a Dx Reasoning environment server running, you can connect directly:

```python
import asyncio

from client import DxReasoningEnv
from models import ActionType, DxAction


async def run() -> None:
    async with DxReasoningEnv(base_url="<ENV_HTTP_URL_HERE>") as env:
        result = await env.reset(task="easy")
        result = await env.step(
            DxAction(action_type=ActionType.ASK_QUESTION, content="What symptoms?")
        )
        print(result.observation.clinical_notes)


asyncio.run(run())
```

Note: When connecting to an existing server, `dx_env.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
import asyncio

from client import DxReasoningEnv
from models import ActionType, DxAction


async def run() -> None:
    async with DxReasoningEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task="medium")
        print(f"Patient: {result.observation.patient_context}")

        for action in [
            DxAction(action_type=ActionType.REQUEST_HISTORY, content="full history"),
            DxAction(action_type=ActionType.ASK_QUESTION, content="chest pain details"),
        ]:
            result = await env.step(action)
            print(f"Reward: {result.reward}")


asyncio.run(run())
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Running Inference with LLM

Use the included inference script to test with LLMs:

```bash
# Set your HuggingFace token
$env:HF_TOKEN = "hf_..."

# Run inference
python inference.py
```

If `HF_TOKEN` is set, the script uses the configured LLM. If `HF_TOKEN` is missing, it runs a deterministic local fallback policy and still emits `[START]`, `[STEP]`, and `[END]` logs with rewards.

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    DxReasoningEnvironment,  # Pass class, not instance
    DxAction,
    DxObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
import asyncio

from client import DxReasoningEnv
from models import ActionType, DxAction


async def run_episode(client_id: int):
    async with DxReasoningEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task="easy")
        for i in range(8):
            action = DxAction(
                action_type=ActionType.ASK_QUESTION,
                content=f"Client {client_id}, step {i}",
            )
            result = await env.step(action)
        return client_id, result.reward


async def main() -> None:
    results = await asyncio.gather(*(run_episode(i) for i in range(4)))
    print(results)


asyncio.run(main())
```

## Grading Model

- Task scoring is deterministic and bounded in the `0.0-1.0` range.
- The explicit grader logic is implemented in `grader.py`.
- Environment step handling calls the grader for:
  - information actions (`request_history`, `ask_question`, `physical_exam`, `order_test`)
  - diagnosis grading (including premature diagnosis penalty for medium/hard)

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From project root
python -c "
from server.dx_reasoning_environment import DxReasoningEnvironment
from models import DxAction, ActionType

env = DxReasoningEnvironment()
obs = env.reset(task='easy')
print(f'Patient: {obs.patient_context}')

step_obs = env.step(DxAction(action_type=ActionType.ASK_QUESTION, content='test'))
print(f'Reward: {step_obs.reward}, Done: {step_obs.done}')
"
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload --port 8000
```

## Project Structure

```
dx_reasoning/
├── .dockerignore         # Docker build exclusions
├── __init__.py           # Module exports
├── README.md             # This file
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependencies (generated)
├── inference.py          # LLM inference script
├── grader.py             # Deterministic task grading logic
├── client.py             # DxReasoningEnv client
├── models.py             # Action, Observation, State models
├── Dockerfile            # Container image definition
└── server/
    ├── __init__.py       # Server module exports
    ├── requirements.txt  # Python dependencies
    ├── app.py            # FastAPI application
    ├── knowledge_base.py # Disease/case library and generators
    └── dx_reasoning_environment.py  # Core environment logic
└── tests/
    └── test_dx_environment.py  # Unit tests
```

## License

This project is licensed under the BSD-style license found in the LICENSE file.