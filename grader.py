from dataclasses import dataclass
from typing import Iterable


@dataclass
class GradeResult:
    score: float
    note: str


class DxGrader:
    """Deterministic grader used by tasks easy/medium/hard."""

    @staticmethod
    def grade_information_action(action_type: str) -> GradeResult:
        base = {
            "request_history": 0.30,
            "ask_question": 0.30,
            "physical_exam": 0.25,
            "order_test": 0.25,
        }.get(action_type, 0.15)
        return GradeResult(score=base, note=f"information_action:{action_type}")

    @staticmethod
    def grade_diagnosis(
        guessed_text: str,
        correct_name: str,
        aliases: Iterable[str],
        task: str,
        evidence_count: int,
    ) -> GradeResult:
        guessed = (guessed_text or "").lower()
        correct = correct_name.lower()
        alias_set = [a.lower() for a in aliases]

        if correct in guessed or any(a in guessed for a in alias_set):
            score = 0.85
            note = "correct"
        else:
            score = 0.40
            note = "incorrect"

        # Penalize premature diagnosis for harder tasks.
        if task in ["medium", "hard"] and evidence_count < 3:
            score -= 0.20
            note += "|premature_penalty"

        return GradeResult(score=max(0.0, min(1.0, score)), note=note)
