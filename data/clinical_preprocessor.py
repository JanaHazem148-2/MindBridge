"""
data/clinical_preprocessor.py
──────────────────────────────
Preprocessing pipeline for clinical text:

1. De-identification (remove or mask PHI)
2. Clinical normalisation (standardise PHQ score text, abbreviations)
3. Quality filtering (remove low-information records)
4. Synthetic data augmentation (paraphrase PHQ assessments for diversity)

This runs as a pre-tokenisation step to enrich the training corpus.
"""

import re
import random
from typing import List, Dict, Any, Iterator


# ── De-identification patterns ────────────────────────────────────────────────
# All DAIC-WOZ participants are already de-identified by the dataset creators.
# These patterns catch any residual PHI that might appear in synthetic additions.

PHI_PATTERNS = [
    # Names
    (re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b"), "[NAME]"),
    # Dates (various formats)
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"), "[DATE]"),
    # Phone numbers
    (re.compile(r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    # SSN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    # Email
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    # Zip codes
    (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "[ZIP]"),
    # MRN (medical record number patterns)
    (re.compile(r"\bMRN[:\s#]*\d{6,10}\b", re.IGNORECASE), "[MRN]"),
]


def deidentify(text: str) -> str:
    """Apply de-identification patterns to text."""
    for pattern, replacement in PHI_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── Clinical normalisation ────────────────────────────────────────────────────

ABBREVIATION_MAP = {
    "PHQ": "Patient Health Questionnaire",
    "PCL-C": "PTSD Checklist – Civilian",
    "CBT": "cognitive behavioral therapy",
    "DBT": "dialectical behavior therapy",
    "MDD": "major depressive disorder",
    "GAD": "generalized anxiety disorder",
    "PTSD": "post-traumatic stress disorder",
    "SI": "suicidal ideation",
    "SH": "self-harm",
    "SA": "suicide attempt",
    "IFS": "Internal Family Systems",
    "ACT": "Acceptance and Commitment Therapy",
    "MI": "motivational interviewing",
    "EBP": "evidence-based practice",
    "TAU": "treatment as usual",
    "RCT": "randomized controlled trial",
}


def expand_abbreviations(text: str) -> str:
    """
    Expand clinical abbreviations on first mention.
    This teaches the model the full forms during pre-training.
    """
    used = set()
    words = text.split()
    result = []
    for word in words:
        clean = word.strip(".,;:()")
        if clean in ABBREVIATION_MAP and clean not in used:
            result.append(f"{clean} ({ABBREVIATION_MAP[clean]})")
            used.add(clean)
        else:
            result.append(word)
    return " ".join(result)


# ── Quality filter ────────────────────────────────────────────────────────────

def is_high_quality(text: str, min_length: int = 50) -> bool:
    """
    Return True if the text is worth including in training.
    Filters out: too short, repetitive, non-informative.
    """
    if len(text) < min_length:
        return False
    if len(set(text.split())) < 10:   # fewer than 10 unique words → not useful
        return False
    # Reject if >50% of chars are non-alphabetic (likely binary/encoding artifact)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False
    return True


# ── Synthetic augmentation ────────────────────────────────────────────────────
# Generate paraphrases of PHQ descriptions to diversify the training corpus.
# These are rule-based (no model needed for pre-training data generation).

PHQ_SEVERITY_TEMPLATES = {
    "minimal": [
        "The patient's PHQ-8 score of {score} falls in the minimal range (0–4), "
        "suggesting little to no depressive symptomatology at this time.",
        "With a PHQ-8 score of {score}, this individual screens negative for clinically "
        "significant depression.",
        "Scores between 0 and 4 on the PHQ-8, such as this patient's score of {score}, "
        "indicate minimal depressive symptoms that are unlikely to require intervention.",
    ],
    "mild": [
        "A PHQ-8 score of {score} indicates mild depressive symptoms. "
        "Watchful waiting and psychoeducation are typically recommended.",
        "This patient scores {score} on the PHQ-8, placing them in the mild category. "
        "Follow-up assessment in 4–6 weeks is advised.",
        "Mild depression (PHQ-8 = {score}) may benefit from behavioral activation "
        "and lifestyle interventions before initiating pharmacotherapy.",
    ],
    "moderate": [
        "A PHQ-8 score of {score} indicates moderate depression. "
        "This warrants a formal clinical assessment and consideration of treatment.",
        "The patient's score of {score} on the PHQ-8 is consistent with moderate "
        "depressive disorder. A combination of psychotherapy and pharmacotherapy "
        "may be indicated.",
        "Moderate depressive symptoms (PHQ-8 = {score}) significantly impact daily "
        "functioning and require a structured treatment plan.",
    ],
    "moderately_severe": [
        "A PHQ-8 of {score} reflects moderately severe depression, associated with "
        "substantial functional impairment. Prompt clinical intervention is indicated.",
        "This score of {score} on the PHQ-8 places the patient in the moderately severe "
        "category, suggesting the need for active treatment including antidepressant "
        "medication and psychotherapy.",
    ],
    "severe": [
        "A PHQ-8 score of {score} indicates severe depression. Urgent evaluation for "
        "safety, hospitalisation need, and intensive treatment is required.",
        "Scores of 20 and above, such as this patient's {score}, indicate severe "
        "depressive disorder. Risk of self-harm should be assessed immediately.",
        "Severe depression (PHQ-8 = {score}) requires urgent clinical attention. "
        "Safety planning and intensive support are essential.",
    ],
}

SEVERITY_THRESHOLDS = [
    (0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"),
    (15, 19, "moderately_severe"), (20, 24, "severe"),
]

def score_to_severity(score: int) -> str:
    for lo, hi, sev in SEVERITY_THRESHOLDS:
        if lo <= score <= hi:
            return sev
    return "severe"


def augment_phq_text(score: int, n_variants: int = 3) -> List[str]:
    """Generate n_variants paraphrase descriptions for a given PHQ-8 score."""
    severity = score_to_severity(score)
    templates = PHQ_SEVERITY_TEMPLATES.get(severity, [])
    selected = random.sample(templates, min(n_variants, len(templates)))
    return [t.format(score=score) for t in selected]


# ── Full preprocessing pipeline ───────────────────────────────────────────────

def preprocess_record(record: Dict[str, Any], augment: bool = True) -> List[str]:
    """
    Apply full preprocessing pipeline to one record.
    Returns a list of text variants (1 original + N augmented).
    """
    text = record["text"]

    # Step 1: de-identify
    text = deidentify(text)

    # Step 2: quality check
    if not is_high_quality(text):
        return []

    # Step 3: abbreviation expansion (optional — can bloat token count)
    # text = expand_abbreviations(text)  # Uncomment if desired

    variants = [text]

    # Step 4: augmentation for DAIC-WOZ PHQ records
    if augment and record["source"] == "daic_woz":
        phq_score = record["labels"].get("phq_score")
        if phq_score is not None:
            augmented = augment_phq_text(phq_score, n_variants=2)
            variants.extend(augmented)

    return variants


def preprocess_corpus(
    records: Iterator[Dict[str, Any]],
    augment: bool = True,
) -> Iterator[str]:
    """
    Full pipeline: iterate records → preprocess each → yield text strings.
    """
    for record in records:
        for text in preprocess_record(record, augment=augment):
            yield text


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/mindbridge")
    from data.dataset_loader import ClinicalDatasetLoader

    loader = ClinicalDatasetLoader(
        daic_woz_dir="/mnt/user-data/uploads",
        iemocap_zip="/mnt/user-data/uploads/archive__7___1_.zip",
        rse_zip="/mnt/user-data/uploads/archive__8_.zip",
    )

    records = list(loader.iter_records())
    texts = list(preprocess_corpus(iter(records), augment=True))

    print(f"Raw records        : {len(records)}")
    print(f"After preprocessing: {len(texts)}  (augmented)")
    print()

    # Show augmentation example
    print("=== PHQ augmentation example (score=12) ===")
    for variant in augment_phq_text(12, n_variants=3):
        print(f"  • {variant}\n")

    print("=== De-identification test ===")
    phi_text = "Patient John Smith (MRN: 123456789) seen on 03/15/2024. Phone: 555-867-5309"
    print(f"Before: {phi_text}")
    print(f"After : {deidentify(phi_text)}")
