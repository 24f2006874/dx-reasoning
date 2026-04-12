# server/dx_reasoning_environment.py
import uuid
import random
from typing import Optional, Dict, List
try:
    from ..models import DxAction, DxObservation, DxState, ActionType
except ImportError:
    from models import DxAction, DxObservation, DxState, ActionType
try:
    from ..grader import DxGrader
except ImportError:
    from grader import DxGrader
from openenv.core.env_server import Environment
from .knowledge_base import knowledge_base, Symptom

class DxReasoningEnvironment(Environment):  # MUST inherit from Environment
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()  # Required
        self._state: Optional[DxState] = None
        self._patient_profile = None
        self._disease_template = None
        self._history_revealed = []
        self._exam_findings = []
        self._test_results = {}
        self._diagnoses_attempted = []
        self._chief_complaints_selected = []

    def reset(self, task: str = "easy", **kwargs) -> DxObservation:
        if task not in ["easy", "medium", "hard", "expert"]:
            task = "easy"

        # Get available diseases for this difficulty
        disease_templates = knowledge_base.get_diseases_by_difficulty(task)
        if not disease_templates:
            # Fallback to easy if no diseases found
            disease_templates = knowledge_base.get_diseases_by_difficulty("easy")
        
        # Select random disease
        self._disease_template = random.choice(disease_templates)
        
        # Generate patient profile
        self._patient_profile = knowledge_base.generate_patient_profile(
            self._disease_template, 
            seed=hash(f"{task}_{uuid.uuid4()}") % 2**32
        )

        self._state = DxState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task=task,
            correct_diagnosis=self._disease_template.name,
            patient_safe=True,
            tests_ordered=[],
            diagnoses_attempted=[],
            reward_breakdown={},
        )

        # Reset tracking variables
        self._history_revealed = []
        self._exam_findings = []
        self._test_results = {}
        self._diagnoses_attempted = []
        self._chief_complaints_selected = random.sample(
            self._disease_template.chief_complaints,
            min(2, len(self._disease_template.chief_complaints)),
        )

        # Generate initial patient context
        patient_context = self._generate_patient_context()

        return DxObservation(
            done=False,
            reward=0.0,
            patient_context=patient_context,
            clinical_notes="Gather history first. Do not diagnose too early.",
            hints_remaining=self._get_hints_remaining(task),
            test_results=self._test_results,
            exam_findings=self._exam_findings,
            history_details=self._history_revealed,
            metadata={
                "disease_name": self._disease_template.display_name,
                "difficulty": task,
                "patient_age": self._patient_profile.age,
                "patient_gender": self._patient_profile.gender
            },
        )

    def step(self, action: DxAction, timeout_s: Optional[float] = None, **kwargs) -> DxObservation:
        if not self._state:
            raise ValueError("Call reset first")

        self._state.step_count += 1
        reward = 0.15
        done = False
        notes = f"Action: {action.action_type.value}"
        reward_breakdown = {
            "base": 0.15,
            "action": 0.0,
            "evidence_bonus": 0.0,
            "premature_penalty": 0.0,
            "repeat_penalty": 0.0,
            "efficiency_bonus": 0.0,
            "final": 0.0,
        }

        max_steps_by_task = {
            "easy": 12,
            "medium": 12,
            "hard": 15,
            "expert": 18,
        }
        max_steps = max_steps_by_task.get(self._state.task, 12)
        if self._state.step_count >= max_steps:
            done = True
            notes = f"Episode reached max steps ({max_steps}) without final diagnosis"
            reward = max(0.0, min(1.0, reward))
            reward_breakdown["action"] = reward
            reward_breakdown["final"] = reward
            self._state.reward_breakdown = reward_breakdown
            return DxObservation(
                done=done,
                reward=reward,
                patient_context=self._generate_patient_context(),
                clinical_notes=notes,
                hints_remaining=self._get_hints_remaining(self._state.task),
                test_results=self._test_results,
                exam_findings=self._exam_findings,
                history_details=self._history_revealed,
                metadata={
                    "disease_name": self._disease_template.display_name,
                    "difficulty": self._state.task,
                    "patient_age": self._patient_profile.age,
                    "patient_gender": self._patient_profile.gender,
                    "step_count": self._state.step_count,
                    "diagnoses_attempted": self._diagnoses_attempted,
                    "tests_ordered": self._state.tests_ordered,
                    "reward_breakdown": reward_breakdown,
                    "evidence_count": self._evidence_count(),
                    "timeout_s": timeout_s,
                },
            )

        if action.action_type == ActionType.DIAGNOSE:
            # Handle diagnosis submission
            guessed = (action.content or "").lower()
            correct = self._state.correct_diagnosis.lower()
            aliases = [a.lower() for a in self._disease_template.aliases]
            evidence_count = self._evidence_count()
            self._diagnoses_attempted.append(action.content or "None")
            self._state.diagnoses_attempted = list(self._diagnoses_attempted)

            grade = DxGrader.grade_diagnosis(
                guessed_text=guessed,
                correct_name=correct,
                aliases=aliases,
                task=self._state.task,
                evidence_count=evidence_count,
            )

            reward = grade.score
            if grade.note.startswith("correct"):
                notes = f"✅ CORRECT DIAGNOSIS: {self._disease_template.display_name}"
            else:
                notes = f"❌ INCORRECT: You guessed '{action.content or 'None'}'. Correct: {self._disease_template.display_name}"

            if "premature_penalty" in grade.note:
                reward_breakdown["premature_penalty"] = -0.20
                notes += " | Premature diagnosis penalty applied (insufficient workup)."

            done = True
            notes += f" (step {self._state.step_count})"
            reward_breakdown["action"] = reward
        else:
            # Handle information gathering actions
            response = self._handle_information_action(action)
            notes = response["notes"]
            info_grade = DxGrader.grade_information_action(action.action_type.value)
            reward = max(float(response.get("reward", 0.0)), info_grade.score)
            reward_breakdown["action"] = reward

            if action.action_type == ActionType.ORDER_TEST and action.content:
                requested_test = action.content.strip().lower()
                if requested_test in [t.lower() for t in self._state.tests_ordered]:
                    reward -= 0.05
                    reward_breakdown["repeat_penalty"] = -0.05
                    notes += " | Repeated test penalty."

            reward_breakdown["evidence_bonus"] = 0.0
            
            # Efficiency bonus for later steps
            if self._state.step_count >= 7:
                reward += 0.10
                reward_breakdown["efficiency_bonus"] = 0.10

        reward = max(0.0, min(1.0, reward))
        reward_breakdown["final"] = reward
        self._state.reward_breakdown = reward_breakdown

        return DxObservation(
            done=done,
            reward=reward,
            patient_context=self._generate_patient_context(),
            clinical_notes=notes,
            hints_remaining=self._get_hints_remaining(self._state.task),
            test_results=self._test_results,
            exam_findings=self._exam_findings,
            history_details=self._history_revealed,
            metadata={
                "disease_name": self._disease_template.display_name,
                "difficulty": self._state.task,
                "patient_age": self._patient_profile.age,
                "patient_gender": self._patient_profile.gender,
                "step_count": self._state.step_count,
                "diagnoses_attempted": self._diagnoses_attempted,
                "tests_ordered": self._state.tests_ordered,
                "reward_breakdown": reward_breakdown,
                "evidence_count": self._evidence_count(),
                "timeout_s": timeout_s,
            },
        )

    @property
    def state(self) -> DxState:
        return self._state or DxState()

    def _get_hints_remaining(self, task: str) -> int:
        """Get number of hints remaining for difficulty level."""
        hints_map = {
            "easy": 2,
            "medium": 1,
            "hard": 0,
            "expert": 0
        }
        return hints_map.get(task, 0)

    def _generate_patient_context(self) -> str:
        """Generate initial patient presentation context."""
        age = self._patient_profile.age
        gender = self._patient_profile.gender
        occupation = self._patient_profile.occupation
        
        # Keep chief complaints stable across an episode.
        chief_complaints = self._chief_complaints_selected or self._disease_template.chief_complaints[:2]
        
        # Get vital signs
        vitals = self._patient_profile.vital_signs
        temp = vitals.get("temperature", 37.0)
        hr = vitals.get("heart_rate", 75)
        rr = vitals.get("respiratory_rate", 16)
        bp_sys = vitals.get("blood_pressure_systolic", 120)
        bp_dia = vitals.get("blood_pressure_diastolic", 80)
        spo2 = vitals.get("oxygen_saturation", 97)
        
        # Generate context string
        context = f"{age}-year-old {gender} {occupation} presenting with: {', '.join(chief_complaints)}. "
        context += f"Vitals: T {temp}°C, HR {hr}, RR {rr}, BP {bp_sys}/{bp_dia}, SpO2 {spo2}%"
        
        if self._patient_profile.medical_history:
            context += f". PMH: {', '.join(self._patient_profile.medical_history)}"
        
        return context

    def _handle_information_action(self, action: DxAction) -> Dict:
        """Handle information gathering actions and return response."""
        action_type = action.action_type
        content = action.content or ""
        
        if action_type == ActionType.REQUEST_HISTORY:
            return self._handle_history_request(content)
        elif action_type == ActionType.ASK_QUESTION:
            return self._handle_question(content)
        elif action_type == ActionType.PHYSICAL_EXAM:
            return self._handle_physical_exam(content)
        elif action_type == ActionType.ORDER_TEST:
            return self._handle_test_order(content)
        else:
            return {"notes": "Unknown action", "reward": 0.15}

    def _handle_history_request(self, content: str) -> Dict:
        """Handle patient history requests."""
        if "full" in content.lower() or "complete" in content.lower():
            # Reveal all history findings
            new_findings = [s.name for s in self._disease_template.symptoms if random.random() < s.probability]
            self._history_revealed.extend(new_findings)
            notes = f"Patient reports: {', '.join(new_findings)}"
            reward = 0.35  # Bonus for comprehensive history
        else:
            # Reveal one relevant symptom
            relevant_symptoms = [s for s in self._disease_template.symptoms if s.probability > 0.5]
            if relevant_symptoms:
                symptom = random.choice(relevant_symptoms)
                if symptom.name not in self._history_revealed:
                    self._history_revealed.append(symptom.name)
                    notes = f"Patient reports: {symptom.name} ({symptom.severity} severity)"
                else:
                    notes = f"Patient confirms: {symptom.name} (already noted)"
            else:
                notes = "Patient denies additional symptoms"
            reward = 0.30
        
        return {"notes": notes, "reward": reward}

    def _handle_question(self, content: str) -> Dict:
        """Handle specific questions about symptoms."""
        content_lower = content.lower()
        
        # Check if question is about specific symptoms
        symptom_matches = []
        for symptom in self._disease_template.symptoms:
            if symptom.name.lower() in content_lower:
                symptom_matches.append(symptom)
        
        if symptom_matches:
            # Answer about specific symptoms
            symptom = random.choice(symptom_matches)
            if random.random() < symptom.probability:
                notes = f"Patient confirms: {symptom.name} ({symptom.severity} severity)"
                if symptom.name not in self._history_revealed:
                    self._history_revealed.append(symptom.name)
            else:
                notes = f"Patient denies: {symptom.name}"
        else:
            # General question - reveal a random symptom
            available_symptoms = [s for s in self._disease_template.symptoms if s.name not in self._history_revealed]
            if available_symptoms:
                symptom = random.choice(available_symptoms)
                if random.random() < symptom.probability:
                    self._history_revealed.append(symptom.name)
                    notes = f"Patient reports: {symptom.name} ({symptom.severity} severity)"
                else:
                    notes = f"Patient denies: {symptom.name}"
            else:
                notes = "Patient has no additional symptoms to report"
        
        return {"notes": notes, "reward": 0.30}

    def _handle_physical_exam(self, content: str) -> Dict:
        """Handle physical examination requests."""
        content_lower = content.lower()
        
        # Check for specific exam types
        exam_findings = []
        for finding in self._disease_template.physical_exam_findings:
            if finding.name.lower() in content_lower or "general" in content_lower:
                if random.random() < finding.probability:
                    exam_findings.append(f"{finding.name} ({finding.severity} severity)")
                    if finding.name not in self._exam_findings:
                        self._exam_findings.append(finding.name)
        
        if not exam_findings:
            # Default exam - reveal one finding
            available_findings = [f for f in self._disease_template.physical_exam_findings if f.name not in self._exam_findings]
            if available_findings:
                finding = random.choice(available_findings)
                if random.random() < finding.probability:
                    exam_findings.append(f"{finding.name} ({finding.severity} severity)")
                    self._exam_findings.append(finding.name)
        
        if exam_findings:
            notes = f"Physical exam reveals: {', '.join(exam_findings)}"
        else:
            notes = "Physical exam shows no significant findings"
        
        return {"notes": notes, "reward": 0.25}

    def _handle_test_order(self, content: str) -> Dict:
        """Handle diagnostic test orders."""
        content_lower = content.lower()
        
        # Check for specific tests
        test_results = []
        for test in self._disease_template.lab_findings + self._disease_template.imaging_findings:
            if test.name.lower() in content_lower:
                # Generate test result
                result_value = self._generate_test_result(test)
                test_results.append(f"{test.name}: {result_value}")
                self._test_results[test.name] = result_value
        
        if not test_results:
            # Default test - order CBC
            default_test = next((t for t in self._disease_template.lab_findings if "WBC" in t.name), None)
            if default_test:
                result_value = self._generate_test_result(default_test)
                test_results.append(f"{default_test.name}: {result_value}")
                self._test_results[default_test.name] = result_value
                if default_test.name not in self._state.tests_ordered:
                    self._state.tests_ordered.append(default_test.name)
        else:
            for test in self._disease_template.lab_findings + self._disease_template.imaging_findings:
                if test.name in self._test_results and test.name not in self._state.tests_ordered:
                    self._state.tests_ordered.append(test.name)
        
        if test_results:
            notes = f"Test results: {', '.join(test_results)}"
        else:
            notes = "No test results available"
        
        return {"notes": notes, "reward": 0.25}

    def _generate_test_result(self, test: Symptom) -> str:
        """Generate realistic test result values."""
        if "positive" in test.name.lower():
            return "Positive" if random.random() < test.probability else "Negative"
        elif "elevated" in test.name.lower():
            return "Elevated" if random.random() < test.probability else "Normal"
        elif "low" in test.name.lower():
            return "Low" if random.random() < test.probability else "Normal"
        elif "high" in test.name.lower():
            return "High" if random.random() < test.probability else "Normal"
        elif "count" in test.name.lower():
            base_value = 10000 if "WBC" in test.name else 5000
            variation = random.randint(-2000, 5000)
            return f"{base_value + variation}/mm³"
        elif "pressure" in test.name.lower():
            return f"{random.randint(110, 140)}/{random.randint(70, 90)} mmHg"
        elif "saturation" in test.name.lower():
            return f"{random.randint(92, 98)}%"
        else:
            return "Abnormal" if random.random() < test.probability else "Normal"

    def close(self):
        """Required by OpenEnv for cleanup."""
        pass

    def _evidence_count(self) -> int:
        return len(self._history_revealed) + len(self._exam_findings) + len(self._test_results)
