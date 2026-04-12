# server/knowledge_base.py
"""
Comprehensive medical knowledge base for the Dx Reasoning Environment.
Contains disease templates, symptom profiles, and patient generation logic.
"""

import random
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Symptom:
    """Represents a medical symptom with occurrence probability."""
    name: str
    probability: float  # 0.0 to 1.0, likelihood of presence in typical case
    severity: str = "moderate"  # mild, moderate, severe
    category: str = "symptom"  # symptom, sign, lab finding, imaging finding


@dataclass
class DiseaseTemplate:
    """Complete disease template with all clinical information."""
    name: str
    display_name: str
    aliases: List[str]
    difficulty: str  # easy, medium, hard, expert
    
    # Clinical presentation
    chief_complaints: List[str]
    symptoms: List[Symptom]
    physical_exam_findings: List[Symptom]
    lab_findings: List[Symptom]
    imaging_findings: List[Symptom]
    
    # Diagnostic criteria
    essential_findings: List[str]  # Must-have for diagnosis
    supportive_findings: List[str]  # Strengthen diagnosis
    
    # Risk factors
    risk_factors: List[str]
    epidemiology: Dict[str, str]  # age, gender, season, geography
    
    # Differential diagnoses
    differentials: List[str]  # Diseases to distinguish from
    
    # Teaching information
    pathophysiology: str
    key_learning_points: List[str]
    
    # Management hints
    red_flags: List[str]  # Warning signs requiring urgent action
    common_misdiagnoses: List[str]


@dataclass
class PatientProfile:
    """Generated patient demographic and background."""
    age: int
    gender: str
    occupation: str
    medical_history: List[str]
    medications: List[str]
    allergies: List[str]
    social_history: Dict[str, str]  # smoking, alcohol, drugs, travel
    family_history: List[str]
    vital_signs: Dict[str, float]


class MedicalKnowledgeBase:
    """Central repository of medical knowledge for the environment."""
    
    def __init__(self):
        self.disease_templates: Dict[str, DiseaseTemplate] = {}
        self._load_disease_templates()
    
    def _load_disease_templates(self):
        """Load all disease templates into the knowledge base."""
        templates = self._get_all_disease_templates()
        for template in templates:
            self.disease_templates[template.name] = template
    
    def get_disease_template(self, disease_name: str) -> Optional[DiseaseTemplate]:
        """Retrieve a disease template by name."""
        return self.disease_templates.get(disease_name)
    
    def get_diseases_by_difficulty(self, difficulty: str) -> List[DiseaseTemplate]:
        """Get all diseases for a given difficulty level."""
        return [t for t in self.disease_templates.values() if t.difficulty == difficulty]
    
    def get_differential_diagnoses(self, disease_name: str) -> List[str]:
        """Get differential diagnoses for a given disease."""
        template = self.disease_templates.get(disease_name)
        return template.differentials if template else []
    
    def _get_all_disease_templates(self) -> List[DiseaseTemplate]:
        """Define all disease templates."""
        return [
            # ==================== EASY CASES ====================
            DiseaseTemplate(
                name="strep_throat",
                display_name="Streptococcal Pharyngitis",
                aliases=["strep throat", "strep", "streptococcal pharyngitis", "group a strep"],
                difficulty="easy",
                chief_complaints=[
                    "sore throat",
                    "throat pain",
                    "pain when swallowing"
                ],
                symptoms=[
                    Symptom("sudden onset sore throat", 0.95, "severe"),
                    Symptom("pain on swallowing", 0.90, "moderate"),
                    Symptom("fever", 0.85, "moderate"),
                    Symptom("headache", 0.60, "mild"),
                    Symptom("nausea", 0.40, "mild"),
                    Symptom("abdominal pain", 0.35, "mild"),
                    Symptom("runny nose", 0.10, "mild"),  # Usually absent
                    Symptom("cough", 0.10, "mild"),  # Usually absent
                ],
                physical_exam_findings=[
                    Symptom("fever (>38°C)", 0.85, "moderate", "sign"),
                    Symptom("tonsillar exudates", 0.70, "moderate", "sign"),
                    Symptom("swollen tender anterior cervical lymph nodes", 0.80, "moderate", "sign"),
                    Symptom("palatal petechiae", 0.30, "mild", "sign"),
                    Symptom("scarlatiniform rash", 0.15, "mild", "sign"),
                    Symptom("tonsillar erythema", 0.90, "moderate", "sign"),
                    Symptom("absence of cough", 0.85, "mild", "sign"),
                ],
                lab_findings=[
                    Symptom("positive rapid strep test", 0.85, "severe", "lab finding"),
                    Symptom("positive throat culture for GAS", 0.95, "severe", "lab finding"),
                    Symptom("elevated WBC count", 0.70, "moderate", "lab finding"),
                    Symptom("negative monospot test", 0.90, "mild", "lab finding"),
                ],
                imaging_findings=[],
                essential_findings=["sore throat", "fever", "tonsillar exudates or erythema"],
                supportive_findings=["tender anterior cervical adenopathy", "absence of cough", "positive strep test"],
                risk_factors=["age 5-15 years", "close contact with infected person", "crowded settings", "winter/early spring"],
                epidemiology={
                    "age": "most common in children 5-15 years",
                    "gender": "equal distribution",
                    "season": "winter and early spring",
                    "geography": "worldwide"
                },
                differentials=["viral pharyngitis", "infectious mononucleosis", "peritonsillar abscess", "diphtheria"],
                pathophysiology="Group A Streptococcus (Streptococcus pyogenes) infects the pharyngeal mucosa, triggering an inflammatory response with local tissue damage and systemic symptoms.",
                key_learning_points=[
                    "Use Centor criteria for clinical diagnosis",
                    "Always confirm with rapid test or culture before treating",
                    "Untreated strep can lead to rheumatic fever",
                    "Cough and runny nose suggest viral etiology instead"
                ],
                red_flags=["difficulty breathing", "drooling", "trismus", "unilateral swelling"],
                common_misdiagnoses=["viral pharyngitis", "mononucleosis"]
            ),
            
            DiseaseTemplate(
                name="appendicitis",
                display_name="Acute Appendicitis",
                aliases=["appendicitis", "acute appendicitis", "inflamed appendix"],
                difficulty="easy",
                chief_complaints=[
                    "abdominal pain",
                    "right lower quadrant pain",
                    "stomach pain"
                ],
                symptoms=[
                    Symptom("periumbilical pain migrating to RLQ", 0.80, "severe"),
                    Symptom("loss of appetite", 0.85, "moderate"),
                    Symptom("nausea", 0.80, "moderate"),
                    Symptom("vomiting", 0.65, "moderate"),
                    Symptom("low-grade fever", 0.60, "mild"),
                    Symptom("constipation", 0.30, "mild"),
                    Symptom("diarrhea", 0.20, "mild"),
                ],
                physical_exam_findings=[
                    Symptom("McBurney's point tenderness", 0.90, "severe", "sign"),
                    Symptom("rebound tenderness", 0.75, "severe", "sign"),
                    Symptom("guarding", 0.70, "moderate", "sign"),
                    Symptom("Rovsing's sign positive", 0.50, "moderate", "sign"),
                    Symptom("psoas sign positive", 0.40, "moderate", "sign"),
                    Symptom("obturator sign positive", 0.30, "mild", "sign"),
                    Symptom("fever (>38°C)", 0.50, "mild", "sign"),
                ],
                lab_findings=[
                    Symptom("elevated WBC count (>10,000)", 0.85, "moderate", "lab finding"),
                    Symptom("left shift (bandemia)", 0.70, "moderate", "lab finding"),
                    Symptom("elevated CRP", 0.75, "moderate", "lab finding"),
                    Symptom("urinalysis usually normal", 0.80, "mild", "lab finding"),
                ],
                imaging_findings=[
                    Symptom("enlarged appendix on CT (>6mm)", 0.90, "severe", "imaging finding"),
                    Symptom("appendiceal wall thickening", 0.85, "moderate", "imaging finding"),
                    Symptom("fat stranding around appendix", 0.80, "moderate", "imaging finding"),
                    Symptom("appendicolith", 0.30, "mild", "imaging finding"),
                    Symptom("free fluid", 0.40, "mild", "imaging finding"),
                ],
                essential_findings=["abdominal pain", "RLQ tenderness", "anorexia"],
                supportive_findings=["migration of pain", "nausea/vomiting", "elevated WBC", "CT findings"],
                risk_factors=["age 10-30 years", "male gender", "family history", "low fiber diet"],
                epidemiology={
                    "age": "most common 10-30 years",
                    "gender": "slightly more common in males",
                    "season": "no seasonal variation",
                    "geography": "worldwide, more common in developed countries"
                },
                differentials=["gastroenteritis", "mesenteric adenitis", "ovarian cyst", "ectopic pregnancy", "UTI"],
                pathophysiology="Obstruction of the appendiceal lumen (usually by fecalith) leads to bacterial overgrowth, inflammation, and potential perforation.",
                key_learning_points=[
                    "Classic presentation is periumbilical pain migrating to RLQ",
                    "Alvarado score helps risk stratification",
                    "CT scan is diagnostic gold standard in adults",
                    "Delay in treatment risks perforation"
                ],
                red_flags=["rigid abdomen", "high fever", "tachycardia", "peritoneal signs"],
                common_misdiagnoses=["gastroenteritis", "mesenteric adenitis", "ovarian pathology"]
            ),
            
            # ==================== MEDIUM CASES ====================
            DiseaseTemplate(
                name="atypical_mi",
                display_name="Atypical Myocardial Infarction",
                aliases=["heart attack", "myocardial infarction", "MI", "acute coronary syndrome", "ACS"],
                difficulty="medium",
                chief_complaints=[
                    "unusual fatigue",
                    "shortness of breath",
                    "epigastric discomfort",
                    "atypical chest discomfort"
                ],
                symptoms=[
                    Symptom("chest discomfort (atypical)", 0.60, "moderate"),
                    Symptom("shortness of breath", 0.80, "moderate"),
                    Symptom("unusual fatigue", 0.75, "moderate"),
                    Symptom("epigastric pain/discomfort", 0.50, "moderate"),
                    Symptom("nausea", 0.55, "mild"),
                    Symptom("diaphoresis", 0.65, "moderate"),
                    Symptom("lightheadedness", 0.50, "mild"),
                    Symptom("back pain", 0.35, "mild"),
                    Symptom("jaw pain", 0.30, "mild"),
                    Symptom("arm pain (atypical distribution)", 0.40, "mild"),
                ],
                physical_exam_findings=[
                    Symptom("diaphoresis", 0.60, "moderate", "sign"),
                    Symptom("tachycardia", 0.50, "moderate", "sign"),
                    Symptom("hypotension", 0.40, "moderate", "sign"),
                    Symptom("S3 gallop", 0.30, "moderate", "sign"),
                    Symptom("lung crackles", 0.35, "mild", "sign"),
                    Symptom("pallor", 0.45, "mild", "sign"),
                ],
                lab_findings=[
                    Symptom("elevated troponin", 0.95, "severe", "lab finding"),
                    Symptom("elevated CK-MB", 0.85, "moderate", "lab finding"),
                    Symptom("elevated myoglobin", 0.75, "moderate", "lab finding"),
                    Symptom("elevated BNP", 0.50, "mild", "lab finding"),
                ],
                imaging_findings=[
                    Symptom("ECG: ST elevation/depression", 0.70, "severe", "imaging finding"),
                    Symptom("ECG: T wave inversion", 0.60, "moderate", "imaging finding"),
                    Symptom("ECG: new LBBB", 0.20, "severe", "imaging finding"),
                    Symptom("echocardiogram: wall motion abnormality", 0.80, "severe", "imaging finding"),
                ],
                essential_findings=["elevated cardiac biomarkers", "symptoms consistent with ischemia"],
                supportive_findings=["ECG changes", "risk factors", "response to nitroglycerin"],
                risk_factors=["diabetes", "female gender", "elderly", "hypertension", "smoking", "family history"],
                epidemiology={
                    "age": "increases with age, common >55",
                    "gender": "women more likely to present atypically",
                    "season": "slightly more common in winter",
                    "geography": "worldwide"
                },
                differentials=["GERD", "panic attack", "pericarditis", "aortic dissection", "pulmonary embolism"],
                pathophysiology="Plaque rupture in coronary arteries leads to thrombus formation and myocardial ischemia. Atypical presentations are common in women, elderly, and diabetics.",
                key_learning_points=[
                    "Women, elderly, and diabetics often present atypically",
                    "Absence of chest pain does NOT rule out MI",
                    "Always get ECG and troponin in at-risk patients",
                    "Time is muscle - early intervention saves lives"
                ],
                red_flags=["crushing chest pain", "severe shortness of breath", "syncope", "cardiogenic shock"],
                common_misdiagnoses=["indigestion", "anxiety", "musculoskeletal pain"]
            ),
            
            DiseaseTemplate(
                name="pulmonary_embolism",
                display_name="Pulmonary Embolism",
                aliases=["pe", "pulmonary embolism", "lung clot", "pulmonary embolus"],
                difficulty="medium",
                chief_complaints=[
                    "shortness of breath",
                    "chest pain",
                    "difficulty breathing"
                ],
                symptoms=[
                    Symptom("sudden onset dyspnea", 0.85, "severe"),
                    Symptom("pleuritic chest pain", 0.70, "moderate"),
                    Symptom("cough", 0.45, "mild"),
                    Symptom("hemoptysis", 0.25, "moderate"),
                    Symptom("anxiety/sense of doom", 0.50, "mild"),
                    Symptom("lightheadedness", 0.45, "mild"),
                    Symptom("syncope", 0.20, "severe"),
                    Symptom("leg swelling", 0.40, "moderate"),
                ],
                physical_exam_findings=[
                    Symptom("tachypnea", 0.80, "moderate", "sign"),
                    Symptom("tachycardia", 0.70, "moderate", "sign"),
                    Symptom("low oxygen saturation", 0.75, "moderate", "sign"),
                    Symptom("lung exam often normal", 0.60, "mild", "sign"),
                    Symptom("unilateral leg swelling", 0.40, "moderate", "sign"),
                    Symptom("calf tenderness", 0.30, "mild", "sign"),
                    Symptom("neck vein distension", 0.25, "moderate", "sign"),
                ],
                lab_findings=[
                    Symptom("elevated D-dimer", 0.90, "moderate", "lab finding"),
                    Symptom("low PaO2 on ABG", 0.70, "moderate", "lab finding"),
                    Symptom("elevated troponin (in massive PE)", 0.40, "moderate", "lab finding"),
                    Symptom("elevated BNP", 0.50, "mild", "lab finding"),
                ],
                imaging_findings=[
                    Symptom("CT pulmonary angiogram: filling defect", 0.90, "severe", "imaging finding"),
                    Symptom("CXR: often normal or Hampton's hump", 0.40, "mild", "imaging finding"),
                    Symptom("ECG: sinus tachycardia", 0.60, "mild", "imaging finding"),
                    Symptom("ECG: S1Q3T3 pattern", 0.20, "moderate", "imaging finding"),
                    Symptom("V/Q scan: mismatch", 0.80, "moderate", "imaging finding"),
                ],
                essential_findings=["sudden dyspnea", "risk factors for DVT/PE"],
                supportive_findings=["pleuritic chest pain", "elevated D-dimer", "CT findings", "hypoxia"],
                risk_factors=["recent surgery", "immobility", "cancer", "OCP use", "pregnancy", "DVT history"],
                epidemiology={
                    "age": "increases with age",
                    "gender": "slightly more common in women",
                    "season": "no seasonal variation",
                    "geography": "worldwide"
                },
                differentials=["pneumonia", "pneumothorax", "MI", "aortic dissection", "anxiety"],
                pathophysiology="Thrombus (usually from DVT) travels to pulmonary arteries, causing obstruction, increased pulmonary pressure, and ventilation-perfusion mismatch.",
                key_learning_points=[
                    "Use Wells criteria for risk stratification",
                    "D-dimer useful for ruling OUT in low-risk patients",
                    "CT pulmonary angiogram is diagnostic gold standard",
                    "Massive PE can cause sudden death"
                ],
                red_flags=["syncope", "hypotension", "severe hypoxia", "cardiac arrest"],
                common_misdiagnoses=["pneumonia", "anxiety", "musculoskeletal chest pain"]
            ),
            
            # ==================== HARD CASES ====================
            DiseaseTemplate(
                name="myasthenia_gravis",
                display_name="Myasthenia Gravis",
                aliases=["mg", "myasthenia gravis", "autoimmune neuromuscular disorder"],
                difficulty="hard",
                chief_complaints=[
                    "drooping eyelids",
                    "double vision",
                    "muscle weakness",
                    "difficulty swallowing"
                ],
                symptoms=[
                    Symptom("ptosis (worsens with fatigue)", 0.85, "moderate"),
                    Symptom("diplopia", 0.75, "moderate"),
                    Symptom("fatigable muscle weakness", 0.90, "moderate"),
                    Symptom("difficulty chewing", 0.55, "moderate"),
                    Symptom("difficulty swallowing", 0.50, "moderate"),
                    Symptom("slurred speech", 0.45, "mild"),
                    Symptom("neck weakness", 0.40, "mild"),
                    Symptom("limb weakness (proximal)", 0.50, "moderate"),
                    Symptom("shortness of breath", 0.30, "severe"),
                ],
                physical_exam_findings=[
                    Symptom("ptosis (unilateral or bilateral)", 0.85, "moderate", "sign"),
                    Symptom("ophthalmoparesis", 0.75, "moderate", "sign"),
                    Symptom("fatigable weakness on repetitive testing", 0.90, "moderate", "sign"),
                    Symptom("positive ice test", 0.70, "moderate", "sign"),
                    Symptom("normal reflexes", 0.90, "mild", "sign"),
                    Symptom("normal sensation", 0.95, "mild", "sign"),
                    Symptom("no muscle atrophy", 0.85, "mild", "sign"),
                ],
                lab_findings=[
                    Symptom("positive anti-AChR antibodies", 0.85, "severe", "lab finding"),
                    Symptom("positive anti-MuSK antibodies", 0.40, "moderate", "lab finding"),
                    Symptom("positive ice test", 0.70, "moderate", "lab finding"),
                    Symptom("elevated CK usually normal", 0.80, "mild", "lab finding"),
                ],
                imaging_findings=[
                    Symptom("repetitive nerve stimulation: decremental response", 0.80, "severe", "imaging finding"),
                    Symptom("single fiber EMG: increased jitter", 0.95, "severe", "imaging finding"),
                    Symptom("CT chest: thymoma or thymic hyperplasia", 0.40, "moderate", "imaging finding"),
                ],
                essential_findings=["fatigable weakness", "ocular symptoms"],
                supportive_findings=["positive antibodies", "positive ice test", "EMG findings", "response to edrophonium"],
                risk_factors=["female gender (younger onset)", "male gender (older onset)", "other autoimmune diseases"],
                epidemiology={
                    "age": "bimodal: 20-40 (women) and 60-80 (men)",
                    "gender": "women > men in younger group",
                    "season": "no seasonal variation",
                    "geography": "worldwide"
                },
                differentials=["lambert-eaton syndrome", "botulism", "guillain-barre", "ms", "stroke"],
                pathophysiology="Autoantibodies target acetylcholine receptors at the neuromuscular junction, causing impaired neuromuscular transmission and fatigable weakness.",
                key_learning_points=[
                    "Weakness worsens with activity, improves with rest",
                    "Ocular symptoms are often the first manifestation",
                    "Myasthenic crisis is a medical emergency",
                    "Edrophonium (Tensilon) test is diagnostic"
                ],
                red_flags=["respiratory distress", "inability to swallow", "severe weakness"],
                common_misdiagnoses=["stroke", "MS", "chronic fatigue", "depression"]
            ),
            
            # ==================== ADDITIONAL EASY CASES ====================
            DiseaseTemplate(
                name="pneumonia",
                display_name="Community-Acquired Pneumonia",
                aliases=["pneumonia", "lung infection", "chest infection"],
                difficulty="easy",
                chief_complaints=[
                    "cough",
                    "fever",
                    "shortness of breath",
                    "chest pain"
                ],
                symptoms=[
                    Symptom("productive cough", 0.85, "moderate"),
                    Symptom("fever with chills", 0.80, "moderate"),
                    Symptom("shortness of breath", 0.70, "moderate"),
                    Symptom("pleuritic chest pain", 0.60, "moderate"),
                    Symptom("fatigue", 0.65, "mild"),
                    Symptom("confusion (elderly)", 0.40, "moderate"),
                ],
                physical_exam_findings=[
                    Symptom("fever (>38°C)", 0.80, "moderate", "sign"),
                    Symptom("crackles on auscultation", 0.85, "moderate", "sign"),
                    Symptom("egophony", 0.50, "moderate", "sign"),
                    Symptom("increased tactile fremitus", 0.55, "moderate", "sign"),
                    Symptom("tachypnea", 0.70, "moderate", "sign"),
                    Symptom("tachycardia", 0.60, "mild", "sign"),
                ],
                lab_findings=[
                    Symptom("elevated WBC count", 0.80, "moderate", "lab finding"),
                    Symptom("elevated procalcitonin", 0.75, "moderate", "lab finding"),
                    Symptom("low oxygen saturation", 0.65, "moderate", "lab finding"),
                    Symptom("positive sputum culture", 0.50, "moderate", "lab finding"),
                ],
                imaging_findings=[
                    Symptom("CXR: lobar consolidation", 0.80, "severe", "imaging finding"),
                    Symptom("CXR: air bronchograms", 0.60, "moderate", "imaging finding"),
                    Symptom("CXR: pleural effusion", 0.30, "mild", "imaging finding"),
                ],
                essential_findings=["cough", "fever", "lung infiltrate on imaging"],
                supportive_findings=["crackles", "elevated WBC", "sputum production"],
                risk_factors=["age >65 or <2", "smoking", "COPD", "immunocompromise"],
                epidemiology={
                    "age": "bimodal: young children and elderly",
                    "gender": "slightly more common in males",
                    "season": "winter months",
                    "geography": "worldwide"
                },
                differentials=["bronchitis", "pulmonary embolism", "heart failure", "lung cancer"],
                pathophysiology="Infection of lung parenchyma leads to inflammation, alveolar exudate, and impaired gas exchange.",
                key_learning_points=[
                    "CURB-65 score helps determine need for hospitalization",
                    "S. pneumoniae is most common bacterial cause",
                    "Chest X-ray is essential for diagnosis",
                    "Consider atypical pathogens in younger patients"
                ],
                red_flags=["severe hypoxia", "septic shock", "multilobar involvement"],
                common_misdiagnoses=["bronchitis", "viral URI"]
            ),
            
            DiseaseTemplate(
                name="diabetic_ketoacidosis",
                display_name="Diabetic Ketoacidosis",
                aliases=["dka", "diabetic ketoacidosis", "ketoacidosis"],
                difficulty="easy",
                chief_complaints=[
                    "excessive thirst",
                    "frequent urination",
                    "nausea and vomiting",
                    "abdominal pain"
                ],
                symptoms=[
                    Symptom("polyuria", 0.90, "moderate"),
                    Symptom("polydipsia", 0.85, "moderate"),
                    Symptom("nausea/vomiting", 0.80, "moderate"),
                    Symptom("abdominal pain", 0.65, "moderate"),
                    Symptom("weakness/fatigue", 0.75, "mild"),
                    Symptom("blurred vision", 0.50, "mild"),
                    Symptom("fruity breath odor", 0.40, "mild"),
                ],
                physical_exam_findings=[
                    Symptom("tachypnea (Kussmaul breathing)", 0.80, "moderate", "sign"),
                    Symptom("dry mucous membranes", 0.85, "moderate", "sign"),
                    Symptom("poor skin turgor", 0.75, "mild", "sign"),
                    Symptom("tachycardia", 0.70, "moderate", "sign"),
                    Symptom("hypotension", 0.50, "moderate", "sign"),
                    Symptom("fruity breath odor", 0.50, "mild", "sign"),
                    Symptom("altered mental status", 0.40, "moderate", "sign"),
                ],
                lab_findings=[
                    Symptom("hyperglycemia (>250 mg/dL)", 0.95, "severe", "lab finding"),
                    Symptom("metabolic acidosis (pH <7.3)", 0.95, "severe", "lab finding"),
                    Symptom("positive serum ketones", 0.95, "severe", "lab finding"),
                    Symptom("elevated anion gap", 0.95, "severe", "lab finding"),
                    Symptom("low bicarbonate (<18 mEq/L)", 0.90, "moderate", "lab finding"),
                    Symptom("elevated BUN/creatinine", 0.70, "mild", "lab finding"),
                    Symptom("hyperkalemia (total body K+ depletion)", 0.80, "moderate", "lab finding"),
                ],
                imaging_findings=[],
                essential_findings=["hyperglycemia", "ketosis", "metabolic acidosis"],
                supportive_findings=["dehydration", "Kussmaul breathing", "abdominal pain"],
                risk_factors=["Type 1 diabetes", "missed insulin doses", "infection", "new-onset diabetes"],
                epidemiology={
                    "age": "any age, common in young Type 1 diabetics",
                    "gender": "equal distribution",
                    "season": "no seasonal variation",
                    "geography": "worldwide"
                },
                differentials=["hyperosmolar hyperglycemic state", "lactic acidosis", "alcoholic ketoacidosis"],
                pathophysiology="Insulin deficiency leads to hyperglycemia, lipolysis, and ketone body production, causing metabolic acidosis and osmotic diuresis.",
                key_learning_points=[
                    "Triad: hyperglycemia, ketosis, acidosis",
                    "Total body potassium is depleted despite normal/high serum K+",
                    "Infection is common precipitating factor",
                    "Cerebral edema is feared complication in children"
                ],
                red_flags=["altered mental status", "severe acidosis", "cardiac arrhythmias"],
                common_misdiagnoses=["gastroenteritis", "sepsis"]
            ),
            
            # ==================== ADDITIONAL MEDIUM CASES ====================
            DiseaseTemplate(
                name="meningitis",
                display_name="Bacterial Meningitis",
                aliases=["meningitis", "bacterial meningitis", "spinal meningitis"],
                difficulty="medium",
                chief_complaints=[
                    "severe headache",
                    "fever",
                    "stiff neck",
                    "confusion"
                ],
                symptoms=[
                    Symptom("severe headache", 0.90, "severe"),
                    Symptom("fever", 0.85, "moderate"),
                    Symptom("neck stiffness", 0.80, "moderate"),
                    Symptom("photophobia", 0.65, "mild"),
                    Symptom("nausea/vomiting", 0.60, "mild"),
                    Symptom("altered mental status", 0.55, "moderate"),
                    Symptom("rash (petechial/purpuric)", 0.35, "severe"),
                ],
                physical_exam_findings=[
                    Symptom("nuchal rigidity", 0.80, "moderate", "sign"),
                    Symptom("Kernig's sign positive", 0.50, "moderate", "sign"),
                    Symptom("Brudzinski's sign positive", 0.45, "moderate", "sign"),
                    Symptom("fever (>38°C)", 0.85, "moderate", "sign"),
                    Symptom("petechial/purpuric rash", 0.40, "severe", "sign"),
                    Symptom("altered mental status", 0.55, "moderate", "sign"),
                ],
                lab_findings=[
                    Symptom("CSF: elevated opening pressure", 0.85, "moderate", "lab finding"),
                    Symptom("CSF: elevated WBC (neutrophils)", 0.95, "severe", "lab finding"),
                    Symptom("CSF: low glucose", 0.85, "moderate", "lab finding"),
                    Symptom("CSF: elevated protein", 0.90, "moderate", "lab finding"),
                    Symptom("CSF: positive Gram stain", 0.70, "severe", "lab finding"),
                    Symptom("blood: elevated WBC", 0.80, "mild", "lab finding"),
                    Symptom("blood: positive cultures", 0.60, "severe", "lab finding"),
                ],
                imaging_findings=[
                    Symptom("CT head: may be normal or show edema", 0.50, "mild", "imaging finding"),
                    Symptom("CT head: hydrocephalus (late)", 0.20, "moderate", "imaging finding"),
                ],
                essential_findings=["fever", "headache", "neck stiffness", "CSF findings"],
                supportive_findings=["photophobia", "altered mental status", "rash", "positive cultures"],
                risk_factors=["age <5 or >60", "immunocompromise", "crowded living", "no vaccination"],
                epidemiology={
                    "age": "bimodal: infants and elderly",
                    "gender": "slightly more common in males",
                    "season": "winter/spring",
                    "geography": "worldwide, meningitis belt in Africa"
                },
                differentials=["viral meningitis", "encephalitis", "subarachnoid hemorrhage", "migraine"],
                pathophysiology="Bacterial infection of meninges causes inflammation, increased intracranial pressure, and potential brain damage.",
                key_learning_points=[
                    "Classic triad (fever, headache, neck stiffness) present in <50%",
                    "Lumbar puncture is diagnostic but get CT first if concern for ICP",
                    "Empiric antibiotics should NOT be delayed",
                    "Droplet precautions are essential"
                ],
                red_flags=["seizures", "focal neurologic deficits", "shock", "purpuric rash"],
                common_misdiagnoses=["viral syndrome", "migraine", "flu"]
            ),
            
            DiseaseTemplate(
                name="ectopic_pregnancy",
                display_name="Ectopic Pregnancy",
                aliases=["ectopic pregnancy", "tubal pregnancy", "extrauterine pregnancy"],
                difficulty="medium",
                chief_complaints=[
                    "abdominal pain",
                    "vaginal bleeding",
                    "missed period"
                ],
                symptoms=[
                    Symptom("unilateral lower abdominal pain", 0.85, "moderate"),
                    Symptom("vaginal bleeding (light)", 0.75, "mild"),
                    Symptom("missed period/amenorrhea", 0.70, "moderate"),
                    Symptom("shoulder tip pain", 0.30, "moderate"),
                    Symptom("dizziness/syncope", 0.35, "moderate"),
                    Symptom("rectal pressure", 0.25, "mild"),
                ],
                physical_exam_findings=[
                    Symptom("adnexal tenderness", 0.80, "moderate", "sign"),
                    Symptom("cervical motion tenderness", 0.60, "moderate", "sign"),
                    Symptom("adnexal mass", 0.40, "moderate", "sign"),
                    Symptom("abdominal tenderness", 0.75, "moderate", "sign"),
                    Symptom("tachycardia (if ruptured)", 0.50, "moderate", "sign"),
                    Symptom("hypotension (if ruptured)", 0.35, "severe", "sign"),
                ],
                lab_findings=[
                    Symptom("positive pregnancy test (beta-hCG)", 0.95, "severe", "lab finding"),
                    Symptom("beta-hCG lower than expected for dates", 0.70, "moderate", "lab finding"),
                    Symptom("beta-hCG plateau or slow rise", 0.75, "moderate", "lab finding"),
                    Symptom("low hemoglobin (if ruptured)", 0.40, "moderate", "lab finding"),
                    Symptom("progesterone low", 0.60, "mild", "lab finding"),
                ],
                imaging_findings=[
                    Symptom("TVUS: empty uterus with positive pregnancy test", 0.85, "severe", "imaging finding"),
                    Symptom("TVUS: adnexal mass/gestational sac", 0.70, "severe", "imaging finding"),
                    Symptom("TVUS: free fluid in cul-de-sac", 0.50, "moderate", "imaging finding"),
                ],
                essential_findings=["positive pregnancy test", "abdominal pain", "empty uterus on ultrasound"],
                supportive_findings=["vaginal bleeding", "adnexal tenderness", "risk factors"],
                risk_factors=["previous ectopic", "PID", "tubal surgery", "IUD", "smoking", "IVF"],
                epidemiology={
                    "age": "most common 20-40 years",
                    "gender": "females only",
                    "season": "no seasonal variation",
                    "geography": "worldwide"
                },
                differentials=["miscarriage", "appendicitis", "ovarian cyst", "PID"],
                pathophysiology="Implantation of fertilized egg outside uterus (usually fallopian tube) leads to growth until rupture, causing life-threatening hemorrhage.",
                key_learning_points=[
                    "Always consider in any woman of childbearing age with abdominal pain",
                    "Rupture is a surgical emergency",
                    "Serial beta-hCG and ultrasound are diagnostic",
                    "Risk factors present in only 50% of cases"
                ],
                red_flags=["syncope", "severe pain", "hemodynamic instability", "shoulder pain"],
                common_misdiagnoses=["miscarriage", "appendicitis", "ovarian cyst rupture"]
            ),
            
            # ==================== EXPERT CASES ====================
            DiseaseTemplate(
                name="pheochromocytoma",
                display_name="Pheochromocytoma",
                aliases=["pheo", "pheochromocytoma", "adrenal tumor"],
                difficulty="expert",
                chief_complaints=[
                    "episodic headaches",
                    "palpitations",
                    "sweating episodes",
                    "high blood pressure"
                ],
                symptoms=[
                    Symptom("paroxysmal headaches", 0.80, "severe"),
                    Symptom("palpitations", 0.75, "moderate"),
                    Symptom("profuse sweating", 0.70, "moderate"),
                    Symptom("hypertension (paroxysmal or sustained)", 0.90, "severe"),
                    Symptom("anxiety/panic", 0.60, "moderate"),
                    Symptom("tremor", 0.50, "mild"),
                    Symptom("pallor", 0.45, "mild"),
                    Symptom("weight loss", 0.35, "mild"),
                ],
                physical_exam_findings=[
                    Symptom("severe hypertension", 0.90, "severe", "sign"),
                    Symptom("tachycardia", 0.65, "moderate", "sign"),
                    Symptom("pallor", 0.50, "mild", "sign"),
                    Symptom("diaphoresis", 0.60, "moderate", "sign"),
                    Symptom("orthostatic hypotension", 0.40, "mild", "sign"),
                ],
                lab_findings=[
                    Symptom("elevated plasma free metanephrines", 0.95, "severe", "lab finding"),
                    Symptom("elevated 24hr urine catecholamines", 0.90, "severe", "lab finding"),
                    Symptom("elevated 24hr urine metanephrines", 0.90, "severe", "lab finding"),
                    Symptom("hyperglycemia", 0.40, "mild", "lab finding"),
                ],
                imaging_findings=[
                    Symptom("CT/MRI: adrenal mass", 0.85, "severe", "imaging finding"),
                    Symptom("MIBG scan: increased uptake", 0.80, "moderate", "imaging finding"),
                    Symptom("PET scan: avid lesion", 0.70, "moderate", "imaging finding"),
                ],
                essential_findings=["paroxysmal hypertension", "elevated metanephrines"],
                supportive_findings=["classic triad (headache, sweating, tachycardia)", "adrenal mass"],
                risk_factors=["MEN2 syndrome", "VHL disease", "NF1", "family history"],
                epidemiology={
                    "age": "peak 30-50 years",
                    "gender": "equal distribution",
                    "season": "no seasonal variation",
                    "geography": "worldwide"
                },
                differentials=["panic disorder", "essential hypertension", "hyperthyroidism", "carcinoid"],
                pathophysiology="Catecholamine-secreting tumor of adrenal medulla causes episodic or sustained release of epinephrine/norepinephrine.",
                key_learning_points=[
                    "Classic triad: headache, sweating, tachycardia",
                    "Can mimic panic attacks",
                    "Alpha-blockade BEFORE beta-blockade to prevent crisis",
                    "10% rule: 10% bilateral, 10% malignant, 10% extra-adrenal"
                ],
                red_flags=["hypertensive crisis", "cardiac arrhythmias", "stroke"],
                common_misdiagnoses=["panic disorder", "essential hypertension", "menopause"]
            ),
            
            DiseaseTemplate(
                name="addisonian_crisis",
                display_name="Acute Adrenal Insufficiency (Addisonian Crisis)",
                aliases=["addisonian crisis", "adrenal crisis", "acute adrenal insufficiency"],
                difficulty="expert",
                chief_complaints=[
                    "severe weakness",
                    "abdominal pain",
                    "vomiting",
                    "confusion"
                ],
                symptoms=[
                    Symptom("profound weakness", 0.90, "severe"),
                    Symptom("severe abdominal pain", 0.80, "severe"),
                    Symptom("nausea/vomiting", 0.85, "moderate"),
                    Symptom("confusion/altered mental status", 0.60, "moderate"),
                    Symptom("fever", 0.50, "mild"),
                    Symptom("back/leg pain", 0.35, "mild"),
                    Symptom("salt craving (chronic)", 0.30, "mild"),
                ],
                physical_exam_findings=[
                    Symptom("hypotension/shock", 0.90, "severe", "sign"),
                    Symptom("tachycardia", 0.80, "moderate", "sign"),
                    Symptom("fever", 0.50, "mild", "sign"),
                    Symptom("hyperpigmentation (chronic)", 0.60, "mild", "sign"),
                    Symptom("dehydration", 0.75, "moderate", "sign"),
                    Symptom("abdominal tenderness", 0.60, "mild", "sign"),
                ],
                lab_findings=[
                    Symptom("hyponatremia", 0.85, "moderate", "lab finding"),
                    Symptom("hyperkalemia", 0.80, "moderate", "lab finding"),
                    Symptom("hypoglycemia", 0.65, "moderate", "lab finding"),
                    Symptom("elevated BUN/creatinine", 0.60, "mild", "lab finding"),
                    Symptom("low cortisol", 0.95, "severe", "lab finding"),
                    Symptom("elevated ACTH", 0.85, "moderate", "lab finding"),
                    Symptom("metabolic acidosis", 0.50, "mild", "lab finding"),
                ],
                imaging_findings=[
                    Symptom("adrenal CT: may show hemorrhage or atrophy", 0.40, "mild", "imaging finding"),
                ],
                essential_findings=["hypotension", "hyponatremia", "hyperkalemia", "low cortisol"],
                supportive_findings=["abdominal pain", "vomiting", "hyperpigmentation", "history of steroid use"],
                risk_factors=["chronic steroid use", "autoimmune disease", "adrenalectomy", "anticoagulation"],
                epidemiology={
                    "age": "any age",
                    "gender": "slightly more common in women",
                    "season": "no seasonal variation",
                    "geography": "worldwide"
                },
                differentials=["sepsis", "acute abdomen", "MI", "hypovolemic shock"],
                pathophysiology="Acute deficiency of cortisol and aldosterone leads to cardiovascular collapse, electrolyte abnormalities, and metabolic derangement.",
                key_learning_points=[
                    "Medical emergency requiring immediate steroids",
                    "Don't wait for labs if clinical suspicion high",
                    "Give stress-dose steroids before any procedure in known adrenal insufficiency",
                    "Hyponatremia + hyperkalemia = think adrenal"
                ],
                red_flags=["shock", "altered mental status", "severe hypoglycemia"],
                common_misdiagnoses=["sepsis", "gastroenteritis", "acute abdomen"]
            ),
        ]
    
    def generate_patient_profile(self, disease_template: DiseaseTemplate, seed: Optional[int] = None) -> PatientProfile:
        """Generate a realistic patient profile for a given disease."""
        if seed is not None:
            random.seed(seed)
        
        # Get epidemiology-based demographics
        epi = disease_template.epidemiology
        age_range = self._get_age_range_from_epidemiology(epi["age"])
        age = random.randint(age_range[0], age_range[1])
        
        # Gender based on epidemiology
        if "female" in epi["gender"].lower() or "women" in epi["gender"].lower():
            gender = random.choice(["female", "female", "female", "male"])
        elif "male" in epi["gender"].lower() or "men" in epi["gender"].lower():
            gender = random.choice(["male", "male", "male", "female"])
        else:
            gender = random.choice(["male", "female"])
        
        # Generate relevant medical history based on risk factors
        medical_history = []
        if "diabetes" in " ".join(disease_template.risk_factors).lower():
            if random.random() < 0.3:
                medical_history.append("Type 2 Diabetes Mellitus")
        if "hypertension" in " ".join(disease_template.risk_factors).lower():
            if random.random() < 0.4:
                medical_history.append("Hypertension")
        if "smoking" in " ".join(disease_template.risk_factors).lower():
            if random.random() < 0.3:
                medical_history.append("Tobacco use disorder")
        
        # Generate social history
        social_history = {
            "smoking": random.choice(["never", "former", "current"]),
            "alcohol": random.choice(["none", "social", "moderate", "heavy"]),
            "drugs": random.choice(["never", "remote", "current"]),
        }
        
        # Generate vital signs (may be abnormal based on disease)
        vital_signs = self._generate_vital_signs(disease_template)
        
        occupations = ["teacher", "engineer", "nurse", "accountant", "student", "retired", "office worker", "construction worker"]
        
        return PatientProfile(
            age=age,
            gender=gender,
            occupation=random.choice(occupations),
            medical_history=medical_history,
            medications=["none"] if not medical_history else [f"medication for {mh}" for mh in medical_history[:2]],
            allergies=random.choice([["NKDA"], ["penicillin"], ["sulfa"], ["none known"]]),
            social_history=social_history,
            family_history=random.choice([["none"], ["heart disease"], ["diabetes"], ["cancer"]]),
            vital_signs=vital_signs
        )
    
    def _get_age_range_from_epidemiology(self, age_desc: str) -> Tuple[int, int]:
        """Parse age description to get age range."""
        normalized = age_desc.lower().strip()

        # Prefer explicit numeric ranges such as "5-15 years".
        match = re.search(r"(\d{1,2})\s*(?:-|to)\s*(\d{1,2})", normalized)
        if match:
            low = int(match.group(1))
            high = int(match.group(2))
            if low <= high:
                return (max(0, low), min(100, high))

        if "children" in normalized or "pediatric" in normalized:
            return (5, 18)
        elif "adolescent" in normalized or "teen" in normalized:
            return (12, 19)
        elif "young adult" in normalized:
            return (18, 35)
        elif "adult" in normalized:
            return (18, 65)
        elif "elderly" in normalized or "older" in normalized or "geriatric" in normalized:
            return (65, 85)
        else:
            return (18, 70)
    
    def _generate_vital_signs(self, disease_template: DiseaseTemplate) -> Dict[str, float]:
        """Generate vital signs that may be abnormal based on disease."""
        vitals = {
            "temperature": round(random.gauss(37.0, 0.3), 1),
            "heart_rate": round(random.gauss(75, 10), 0),
            "respiratory_rate": round(random.gauss(16, 2), 0),
            "blood_pressure_systolic": round(random.gauss(120, 15), 0),
            "blood_pressure_diastolic": round(random.gauss(80, 10), 0),
            "oxygen_saturation": round(random.gauss(97, 1), 0),
        }
        
        # Adjust based on disease symptoms
        fever_symptoms = [s for s in disease_template.symptoms + disease_template.physical_exam_findings 
                         if "fever" in s.name.lower()]
        if fever_symptoms and random.random() < 0.7:
            vitals["temperature"] = round(random.uniform(38.0, 39.5), 1)
        
        tachycardia_symptoms = [s for s in disease_template.symptoms + disease_template.physical_exam_findings 
                               if "tachycardia" in s.name.lower()]
        if tachycardia_symptoms and random.random() < 0.6:
            vitals["heart_rate"] = round(random.uniform(100, 120), 0)
        
        hypoxia_symptoms = [s for s in disease_template.symptoms + disease_template.physical_exam_findings 
                           if "oxygen" in s.name.lower() or "hypoxia" in s.name.lower()]
        if hypoxia_symptoms and random.random() < 0.5:
            vitals["oxygen_saturation"] = round(random.uniform(88, 94), 0)
        
        return vitals


# Singleton instance for global access
knowledge_base = MedicalKnowledgeBase()