# models.py
from pydantic import Field
from typing import Literal, Optional, List, Dict, Any
from enum import Enum
from openenv.core.env_server.interfaces import Action, Observation, State

class ActionType(str, Enum):
    REQUEST_HISTORY = "request_history"
    ASK_QUESTION = "ask_question"
    PHYSICAL_EXAM = "physical_exam"
    ORDER_TEST = "order_test"
    DIAGNOSE = "diagnose"

class DxAction(Action):
    action_type: ActionType
    content: Optional[str] = Field(None, description="Question, test name, exam type, or diagnosis")

class DxObservation(Observation):
    done: bool = False
    reward: float = 0.0
    patient_context: str = ""
    test_results: Dict[str, str] = Field(default_factory=dict)
    exam_findings: List[str] = Field(default_factory=list)
    history_details: List[str] = Field(default_factory=list)
    clinical_notes: str = ""
    hints_remaining: int = 0
    warning: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)   # <-- ADD THIS

class DxState(State):
    episode_id: str = ""
    step_count: int = 0
    task: str = "easy"
    correct_diagnosis: str = ""
    tests_ordered: List[str] = Field(default_factory=list)
    diagnoses_attempted: List[str] = Field(default_factory=list)
    patient_safe: bool = True
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)