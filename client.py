# client.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dx Reasoning Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import DxAction, DxObservation, DxState
except ImportError:
    from models import DxAction, DxObservation, DxState


class DxReasoningEnv(EnvClient[DxAction, DxObservation, DxState]):
    """
    Client for the Dx Reasoning Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with DxReasoningEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task="easy")
        ...     print(result.observation.patient_context)
        ...
        ...     result = client.step(DxAction(action_type="request_history", content="full history"))
        ...     print(result.observation.clinical_notes)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = DxReasoningEnv.from_docker_image("dx_reasoning:latest")
        >>> try:
        ...     result = client.reset(task="medium")
        ...     result = client.step(DxAction(action_type="ask_question", content="When did pain start?"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DxAction) -> Dict:
        """
        Convert DxAction to JSON payload for step message.

        Args:
            action: DxAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type.value,
            "content": action.content,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DxObservation]:
        """
        Parse server response into StepResult[DxObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DxObservation
        """
        obs_data = payload.get("observation", {})
        observation = DxObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            patient_context=obs_data.get("patient_context", ""),
            test_results=obs_data.get("test_results", {}),
            exam_findings=obs_data.get("exam_findings", []),
            history_details=obs_data.get("history_details", []),
            clinical_notes=obs_data.get("clinical_notes", ""),
            hints_remaining=obs_data.get("hints_remaining", 0),
            warning=obs_data.get("warning", ""),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> DxState:
        """
        Parse server response into DxState object.

        Args:
            payload: JSON response from state request

        Returns:
            DxState object with episode_id, step_count, and other fields
        """
        return DxState(**payload)