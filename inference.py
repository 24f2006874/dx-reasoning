# inference.py
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List

from openai import AsyncOpenAI

# Allow running `python inference.py` from this repo root while using package-prefixed imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dx_reasoning.client import DxReasoningEnv
from dx_reasoning.models import ActionType, DxAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")


def _format_rewards(rewards: List[float]) -> str:
    rounded = [round(float(r), 2) for r in rewards]
    return json.dumps(rounded, separators=(",", ":"))


def _safe_action(raw_json: str) -> DxAction:
    try:
        data = json.loads(raw_json)
        action_type = ActionType(data["action_type"])
        content = data.get("content", "")
        return DxAction(action_type=action_type, content=content)
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return DxAction(action_type=ActionType.REQUEST_HISTORY, content="full history")


def _fallback_action(step: int) -> DxAction:
    # Deterministic local policy when no LLM token is available.
    plan = {
        1: DxAction(action_type=ActionType.REQUEST_HISTORY, content="full history"),
        2: DxAction(action_type=ActionType.ASK_QUESTION, content="fever and duration"),
        3: DxAction(action_type=ActionType.PHYSICAL_EXAM, content="general exam"),
        4: DxAction(action_type=ActionType.ORDER_TEST, content="WBC count"),
        5: DxAction(action_type=ActionType.DIAGNOSE, content="strep throat"),
    }
    return plan.get(step, DxAction(action_type=ActionType.DIAGNOSE, content="pneumonia"))


async def main() -> None:
    llm = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

    async with DxReasoningEnv(base_url=ENV_BASE_URL) as env:
        for task in ["easy", "medium", "hard"]:
            model_name = MODEL_NAME if llm else "local_fallback_policy"
            print(f"[START] task={task} env=dx_reasoning model={model_name}")

            reset_result = await env.reset(task=task)
            obs = reset_result.observation
            done = bool(reset_result.done)
            rewards: List[float] = []
            if reset_result.reward is not None:
                rewards.append(float(reset_result.reward))

            step = 0
            while not done and step < 12:
                step += 1
                prompt = (
                    f"Patient: {obs.patient_context}\n"
                    f"Notes: {obs.clinical_notes}\n"
                    f"History: {', '.join(obs.history_details) if obs.history_details else 'none'}\n"
                    f"Exam: {', '.join(obs.exam_findings) if obs.exam_findings else 'none'}\n"
                    f"Tests: {json.dumps(obs.test_results, sort_keys=True) if obs.test_results else '{}'}\n\n"
                    "Choose exactly one action_type from: request_history, ask_question, physical_exam, order_test, diagnose. "
                    "Return JSON only with keys action_type and content."
                )

                if llm:
                    response = await llm.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=100,
                    )
                    raw_content = response.choices[0].message.content or ""
                    action = _safe_action(raw_content)
                else:
                    action = _fallback_action(step)
                step_result = await env.step(action)

                obs = step_result.observation
                done = bool(step_result.done)
                step_reward = float(step_result.reward or 0.0)
                rewards.append(step_reward)

                print(
                    f"[STEP] step={step} action={action.action_type.value} "
                    f"reward={step_reward:.2f} done={str(done).lower()} error=null"
                )

            score = (sum(rewards) / len(rewards)) if rewards else 0.0
            success = str(score >= 0.5).lower()
            print(
                f"[END] success={success} steps={step} score={score:.3f} rewards={_format_rewards(rewards)}"
            )


if __name__ == "__main__":
    asyncio.run(main())