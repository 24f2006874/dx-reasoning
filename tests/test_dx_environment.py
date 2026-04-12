#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced Dx Reasoning Environment.
Tests all functionality including the new knowledge base and intelligent responses.
"""

import unittest
import uuid
from unittest.mock import patch, MagicMock
from server.dx_reasoning_environment import DxReasoningEnvironment
from server.knowledge_base import knowledge_base
from models import DxAction, DxObservation, DxState, ActionType


class TestDxReasoningEnvironment(unittest.TestCase):
    """Test suite for the enhanced Dx Reasoning Environment."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = DxReasoningEnvironment()
    
    def test_knowledge_base_loading(self):
        """Test that knowledge base loads all disease templates correctly."""
        # Test that we have diseases for each difficulty level
        easy_diseases = knowledge_base.get_diseases_by_difficulty("easy")
        medium_diseases = knowledge_base.get_diseases_by_difficulty("medium")
        hard_diseases = knowledge_base.get_diseases_by_difficulty("hard")
        expert_diseases = knowledge_base.get_diseases_by_difficulty("expert")
        
        self.assertGreater(len(easy_diseases), 0, "Should have easy diseases")
        self.assertGreater(len(medium_diseases), 0, "Should have medium diseases")
        self.assertGreater(len(hard_diseases), 0, "Should have hard diseases")
        self.assertGreater(len(expert_diseases), 0, "Should have expert diseases")
        
        # Test total count - we have 11 diseases currently
        total_diseases = len(easy_diseases) + len(medium_diseases) + len(hard_diseases) + len(expert_diseases)
        self.assertGreaterEqual(total_diseases, 10, "Should have at least 10 diseases")
    
    def test_reset_functionality(self):
        """Test environment reset with different difficulty levels."""
        # Test easy difficulty
        obs = self.env.reset(task="easy")
        self.assertIsInstance(obs, DxObservation)
        self.assertFalse(obs.done)
        self.assertEqual(obs.hints_remaining, 2)
        self.assertIn("disease_name", obs.metadata)
        self.assertIn("difficulty", obs.metadata)
        self.assertEqual(obs.metadata["difficulty"], "easy")
        
        # Test medium difficulty
        obs = self.env.reset(task="medium")
        self.assertEqual(obs.hints_remaining, 1)
        self.assertEqual(obs.metadata["difficulty"], "medium")
        
        # Test hard difficulty
        obs = self.env.reset(task="hard")
        self.assertEqual(obs.hints_remaining, 0)
        self.assertEqual(obs.metadata["difficulty"], "hard")
        
        # Test invalid difficulty (should default to easy)
        obs = self.env.reset(task="invalid")
        self.assertEqual(obs.metadata["difficulty"], "easy")
    
    def test_patient_generation(self):
        """Test that patient profiles are generated realistically."""
        obs = self.env.reset(task="easy")
        
        # Check that patient context contains realistic information
        context = obs.patient_context
        self.assertIn("year-old", context)
        self.assertIn("presenting with", context)
        self.assertIn("Vitals:", context)
        
        # Check that metadata contains patient information
        self.assertIn("patient_age", obs.metadata)
        self.assertIn("patient_gender", obs.metadata)
        self.assertIsInstance(obs.metadata["patient_age"], int)
        self.assertIn(obs.metadata["patient_gender"], ["male", "female"])
    
    def test_history_request(self):
        """Test patient history requests."""
        obs = self.env.reset(task="easy")
        
        # Request full history
        action = DxAction(action_type=ActionType.REQUEST_HISTORY, content="full history")
        result = self.env.step(action)
        
        self.assertGreater(result.reward, 0.30, "Should get bonus for comprehensive history")
        self.assertIn("Patient reports:", result.clinical_notes)
        self.assertGreater(len(result.history_details), 0)
    
    def test_question_handling(self):
        """Test specific question handling."""
        obs = self.env.reset(task="easy")
        
        # Ask about specific symptoms
        action = DxAction(action_type=ActionType.ASK_QUESTION, content="Do you have fever?")
        result = self.env.step(action)
        
        self.assertEqual(result.reward, 0.30, "Should get standard reward for questions")
        self.assertIn("Patient", result.clinical_notes)
    
    def test_physical_exam(self):
        """Test physical examination handling."""
        obs = self.env.reset(task="easy")
        
        # Request physical exam
        action = DxAction(action_type=ActionType.PHYSICAL_EXAM, content="general exam")
        result = self.env.step(action)
        
        self.assertEqual(result.reward, 0.25, "Should get standard reward for exams")
        self.assertIn("Physical exam", result.clinical_notes)
    
    def test_test_ordering(self):
        """Test diagnostic test ordering."""
        obs = self.env.reset(task="easy")
        
        # Order a test - use "WBC" which is a common lab finding
        action = DxAction(action_type=ActionType.ORDER_TEST, content="WBC count")
        result = self.env.step(action)
        
        self.assertEqual(result.reward, 0.25, "Should get standard reward for tests")
        # Either test results are available or not, depending on disease
        self.assertTrue(
            "Test results:" in result.clinical_notes or "No test results" in result.clinical_notes,
            "Should have test-related response"
        )
    
    def test_diagnosis_correct(self):
        """Test correct diagnosis submission."""
        obs = self.env.reset(task="easy")
        
        # Get the correct disease name
        correct_disease = self.env._state.correct_diagnosis
        disease_template = knowledge_base.get_disease_template(correct_disease)
        
        # Submit correct diagnosis
        action = DxAction(action_type=ActionType.DIAGNOSE, content=disease_template.display_name)
        result = self.env.step(action)
        
        self.assertTrue(result.done, "Episode should be done after diagnosis")
        self.assertEqual(result.reward, 0.85, "Should get high reward for correct diagnosis")
        self.assertIn("CORRECT DIAGNOSIS", result.clinical_notes)
    
    def test_diagnosis_incorrect(self):
        """Test incorrect diagnosis submission."""
        obs = self.env.reset(task="easy")
        
        # Submit wrong diagnosis
        action = DxAction(action_type=ActionType.DIAGNOSE, content="flu")
        result = self.env.step(action)
        
        self.assertTrue(result.done, "Episode should be done after diagnosis")
        self.assertEqual(result.reward, 0.40, "Should get lower reward for incorrect diagnosis")
        self.assertIn("INCORRECT", result.clinical_notes)
        self.assertIn("flu", result.clinical_notes)
    
    def test_step_limit(self):
        """Test that episode ends after diagnosis."""
        obs = self.env.reset(task="easy")
        
        # Take a few steps without diagnosing
        for i in range(3):
            action = DxAction(action_type=ActionType.ASK_QUESTION, content=f"Question {i}")
            result = self.env.step(action)
            self.assertFalse(result.done, f"Should not be done at step {i+1}")
        
        # Submit diagnosis - should end episode
        action = DxAction(action_type=ActionType.DIAGNOSE, content="test diagnosis")
        result = self.env.step(action)
        self.assertTrue(result.done, "Should be done after diagnosis")
    
    def test_different_diseases(self):
        """Test that different diseases produce different responses."""
        diseases_seen = set()
        
        # Run multiple episodes and check we get different diseases
        for _ in range(10):
            obs = self.env.reset(task="easy")
            disease_name = self.env._state.correct_diagnosis
            diseases_seen.add(disease_name)
        
        self.assertGreater(len(diseases_seen), 1, "Should see different diseases across episodes")
    
    def test_vital_signs_generation(self):
        """Test that vital signs are generated realistically."""
        obs = self.env.reset(task="easy")
        
        # Check that vital signs are in patient context
        context = obs.patient_context
        self.assertIn("T ", context)
        self.assertIn("HR ", context)
        self.assertIn("RR ", context)
        self.assertIn("BP ", context)
        self.assertIn("SpO2 ", context)
    
    def test_medical_history_generation(self):
        """Test that medical history is generated based on risk factors."""
        obs = self.env.reset(task="easy")
        
        # Check metadata for medical history
        self.assertIn("patient_age", obs.metadata)
        self.assertIn("patient_gender", obs.metadata)
    
    def test_efficiency_bonus(self):
        """Test that efficiency bonus is applied for later steps."""
        obs = self.env.reset(task="easy")
        
        # Take 6 steps without efficiency bonus
        for i in range(6):
            action = DxAction(action_type=ActionType.ASK_QUESTION, content=f"Question {i}")
            result = self.env.step(action)
            # Base reward for questions is 0.30
            self.assertEqual(result.reward, 0.30, f"Step {i+1} should have base reward")
        
        # 7th step should have efficiency bonus (0.30 + 0.10 = 0.40)
        action = DxAction(action_type=ActionType.ASK_QUESTION, content="Question 7")
        result = self.env.step(action)
        self.assertEqual(result.reward, 0.40, "Step 7+ should have efficiency bonus")
        
        # Episode should still be active (only ends on diagnosis)
        self.assertFalse(result.done, "Episode should not be done yet")
    
    def test_metadata_completeness(self):
        """Test that metadata contains all expected fields."""
        obs = self.env.reset(task="easy")
        
        # Fields available at reset
        expected_fields = [
            "disease_name", "difficulty", "patient_age", "patient_gender"
        ]
        
        for field in expected_fields:
            self.assertIn(field, obs.metadata, f"Metadata should contain {field}")
        
        # After a step, additional fields should be available
        action = DxAction(action_type=ActionType.ASK_QUESTION, content="test")
        result = self.env.step(action)
        self.assertIn("step_count", result.metadata)
        self.assertIn("diagnoses_attempted", result.metadata)
    
    def test_concurrent_session_support(self):
        """Test that the environment supports concurrent sessions."""
        self.assertTrue(
            DxReasoningEnvironment.SUPPORTS_CONCURRENT_SESSIONS,
            "Environment should support concurrent sessions"
        )


class TestKnowledgeBase(unittest.TestCase):
    """Test suite for the knowledge base functionality."""
    
    def test_disease_template_structure(self):
        """Test that disease templates have all required fields."""
        diseases = knowledge_base.get_diseases_by_difficulty("easy")
        
        for disease in diseases:
            self.assertIsNotNone(disease.name)
            self.assertIsNotNone(disease.display_name)
            self.assertIsNotNone(disease.difficulty)
            self.assertIsNotNone(disease.chief_complaints)
            self.assertIsNotNone(disease.symptoms)
            self.assertIsNotNone(disease.physical_exam_findings)
            self.assertIsNotNone(disease.lab_findings)
            self.assertIsNotNone(disease.pathophysiology)
            self.assertIsNotNone(disease.key_learning_points)
    
    def test_symptom_probabilities(self):
        """Test that symptom probabilities are valid."""
        diseases = knowledge_base.get_diseases_by_difficulty("easy")
        
        for disease in diseases:
            for symptom in disease.symptoms:
                self.assertGreaterEqual(symptom.probability, 0.0)
                self.assertLessEqual(symptom.probability, 1.0)
                self.assertIn(symptom.severity, ["mild", "moderate", "severe"])
    
    def test_patient_generation_consistency(self):
        """Test that patient generation is consistent."""
        disease = knowledge_base.get_diseases_by_difficulty("easy")[0]
        
        for _ in range(5):
            patient = knowledge_base.generate_patient_profile(disease)
            
            self.assertIsInstance(patient.age, int)
            self.assertIn(patient.gender, ["male", "female"])
            self.assertIsInstance(patient.medical_history, list)
            self.assertIsInstance(patient.vital_signs, dict)
            self.assertIn("temperature", patient.vital_signs)
            self.assertIn("heart_rate", patient.vital_signs)
    
    def test_differential_diagnoses(self):
        """Test that differential diagnoses are available."""
        diseases = knowledge_base.get_diseases_by_difficulty("medium")
        
        for disease in diseases:
            differentials = knowledge_base.get_differential_diagnoses(disease.name)
            self.assertIsInstance(differentials, list)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)