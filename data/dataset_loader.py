"""
data/dataset_loader.py
──────────────────────
Loads and normalises all MindBridge training datasets into a unified schema.

Unified record schema
─────────────────────
{
    "text":        str,          # The actual text fed to the LLM
    "source":      str,          # Dataset origin tag
    "participant": Optional[str],# De-identified ID
    "labels": {
        "phq_score":      Optional[int],    # 0-24
        "phq_binary":     Optional[int],    # 0=no depression, 1=depression
        "phq_items":      Optional[dict],   # 8 individual PHQ item scores
        "ptsd_score":     Optional[int],
        "ptsd_binary":    Optional[int],
        "depression_severity": Optional[str],  # minimal/mild/moderate/severe
        "emotions":       Optional[List[str]], # IEMOCAP labels
    },
    "split": str,                # train | dev | test
    "safety_flag": bool,         # True = contains crisis content → restricted use
}
"""

import os
import zipfile
import io
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List
import pandas as pd


# ── PHQ-8 scoring helpers ─────────────────────────────────────────────────────

PHQ_ITEM_NAMES = [
    "NoInterest", "Depressed", "Sleep", "Tired",
    "Appetite", "Failure", "Concentration", "Psychomotor"
]

PHQ_SEVERITY_THRESHOLDS = [
    (0, 4,  "minimal"),
    (5, 9,  "mild"),
    (10, 14, "moderate"),
    (15, 19, "moderately_severe"),
    (20, 24, "severe"),
]

def phq_severity(score: int) -> str:
    for lo, hi, label in PHQ_SEVERITY_THRESHOLDS:
        if lo <= score <= hi:
            return label
    return "unknown"


# ── Clinical context templates ────────────────────────────────────────────────
# These turn structured PHQ data into natural language pre-training text.
# The model learns to generate and reason about these patterns.

def phq_to_text(row: pd.Series, source_name: str) -> str:
    """
    Convert a PHQ-8 assessment row into a training text snippet.
    Format: structured clinical note + conversational summary.
    """
    items = {}
    for col in row.index:
        for item in PHQ_ITEM_NAMES:
            if item.lower() in col.lower() and pd.notna(row[col]):
                items[item] = int(row[col])

    score_col = next(
        (c for c in row.index if "total" in c.lower() or "score" in c.lower()), None
    )
    score = int(row[score_col]) if score_col and pd.notna(row.get(score_col, None)) else sum(items.values())
    severity = phq_severity(score)
    gender = row.get("gender", row.get("Gender", "unknown"))

    # Build item description
    item_lines = []
    item_descriptions = {
        "NoInterest":    ("little interest or pleasure in doing things",    [0,1,2,3]),
        "Depressed":     ("feeling down, depressed, or hopeless",           [0,1,2,3]),
        "Sleep":         ("trouble falling or staying asleep, or sleeping too much", [0,1,2,3]),
        "Tired":         ("feeling tired or having little energy",          [0,1,2,3]),
        "Appetite":      ("poor appetite or overeating",                    [0,1,2,3]),
        "Failure":       ("feeling bad about yourself — a failure",         [0,1,2,3]),
        "Concentration": ("trouble concentrating on things",                [0,1,2,3]),
        "Psychomotor":   ("moving or speaking slowly, or being fidgety/restless", [0,1,2,3]),
    }
    freq_labels = ["not at all", "several days", "more than half the days", "nearly every day"]

    for item, val in items.items():
        if item in item_descriptions:
            desc, _ = item_descriptions[item]
            freq = freq_labels[val] if val < len(freq_labels) else str(val)
            item_lines.append(f"  - {desc}: {freq} (score {val})")

    pronoun = "They" if str(gender).lower() not in ["male","m"] else "He"
    pronoun = "She" if str(gender).lower() in ["female","f"] else pronoun

    text = f"""[CLINICAL ASSESSMENT — PHQ-8]
Patient gender: {gender}
PHQ-8 Total Score: {score}/24 — {severity.replace('_', ' ')} depression

Symptom breakdown:
{chr(10).join(item_lines) if item_lines else '  (item-level data not available)'}

Clinical summary: {pronoun} reports a PHQ-8 score of {score}, indicating {severity.replace('_', ' ')} \
depressive symptoms. {'This score meets criteria for a likely major depressive episode.' if score >= 10 else \
'This score does not meet the threshold for a major depressive episode at this time.'}
[END ASSESSMENT]"""
    return text


def ptsd_to_text(row: pd.Series) -> str:
    """Convert PCL-C PTSD data to training text."""
    ptsd_score = row.get("PTSD_severity", row.get("PCL-C (PTSD)", None))
    ptsd_binary = row.get("PTSD_label", row.get("PCL-C (PTSD)", None))
    if ptsd_score is None:
        return ""

    ptsd_score = int(ptsd_score) if pd.notna(ptsd_score) else 0
    severity = "subclinical" if ptsd_score < 44 else ("moderate" if ptsd_score < 60 else "severe")

    text = f"""[CLINICAL ASSESSMENT — PCL-C PTSD CHECKLIST]
PCL-C Total Score: {ptsd_score}/85 — {severity} PTSD symptom range
{'PTSD diagnosis indicated (score ≥ 44).' if ptsd_score >= 44 else 'Score below diagnostic threshold.'}
[END ASSESSMENT]"""
    return text


# ── Dataset loaders ───────────────────────────────────────────────────────────

class DAICWOZLoader:
    """
    Loads the DAIC-WOZ depression dataset splits.
    Files: train_split.csv, dev_split.csv, test_split.csv
    Augmented with item-level labels from detailed_lables.csv and Detailed_PHQ8_Labels.csv
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> Iterator[Dict[str, Any]]:
        # Load split files
        splits = {}
        for split_name in ["train", "dev", "test"]:
            path = self.data_dir / f"{split_name}_split.csv"
            if path.exists():
                splits[split_name] = pd.read_csv(path)

        # Load detailed item-level labels
        detailed_path = self.data_dir / "detailed_lables.csv"
        phq8_path = self.data_dir / "Detailed_PHQ8_Labels.csv"

        detailed_df = pd.read_csv(detailed_path) if detailed_path.exists() else None
        phq8_df = pd.read_csv(phq8_path) if phq8_path.exists() else None

        # Build participant → item scores lookup
        item_lookup: Dict[int, dict] = {}
        if phq8_df is not None:
            for _, row in phq8_df.iterrows():
                pid = int(row["Participant_ID"])
                item_lookup[pid] = {
                    item: int(row.get(f"PHQ_8{item}", 0))
                    for item in PHQ_ITEM_NAMES
                    if f"PHQ_8{item}" in row.index
                }

        for split_name, df in splits.items():
            for _, row in df.iterrows():
                pid = int(row["Participant_ID"])
                phq_score = int(row["PHQ_Score"])
                phq_binary = int(row["PHQ_Binary"])
                gender = row["Gender"]
                ptsd_raw = row.get("PTSD Severity", 0)
                ptsd_score = int(ptsd_raw) if pd.notna(ptsd_raw) else 0

                # Synthesise item scores from lookup if available
                phq_items = item_lookup.get(pid, {})

                # Build training text from PHQ items
                phq_row = pd.Series({
                    **{f"PHQ_{k}": v for k, v in phq_items.items()},
                    "PHQ_Score": phq_score,
                    "gender": gender,
                })
                phq_text = phq_to_text(phq_row, "daic_woz")
                ptsd_text = ptsd_to_text(pd.Series({"PTSD_severity": ptsd_score}))

                combined_text = phq_text
                if ptsd_text:
                    combined_text += "\n\n" + ptsd_text

                # Safety flag: high PHQ or high PTSD → restrict generation use
                safety_flag = phq_score >= 15 or ptsd_score >= 60

                yield {
                    "text": combined_text,
                    "source": "daic_woz",
                    "participant": str(pid),
                    "labels": {
                        "phq_score": phq_score,
                        "phq_binary": phq_binary,
                        "phq_items": phq_items,
                        "ptsd_score": ptsd_score,
                        "ptsd_binary": 1 if ptsd_score >= 44 else 0,
                        "depression_severity": phq_severity(phq_score),
                        "emotions": None,
                    },
                    "split": split_name,
                    "safety_flag": safety_flag,
                }


class IEMOCAPLoader:
    """
    Loads the IEMOCAP emotion dataset.
    Source: archive__7___1_.zip → iemocap_full_dataset.csv
    Columns: session, method, gender, emotion, n_annotators, agreement, path
    """

    EMOTION_MAP = {
        "neu": "neutral",
        "hap": "happy",
        "sad": "sadness",
        "ang": "anger",
        "fru": "frustration",
        "exc": "excited",
        "fea": "fear",
        "sur": "surprise",
        "dis": "disgust",
        "oth": "other",
    }

    def __init__(self, zip_path: str):
        self.zip_path = zip_path

    def load(self) -> Iterator[Dict[str, Any]]:
        with zipfile.ZipFile(self.zip_path) as z:
            with z.open("iemocap_full_dataset.csv") as f:
                df = pd.read_csv(f)

        for _, row in df.iterrows():
            emotion_code = str(row["emotion"]).lower()
            emotion_label = self.EMOTION_MAP.get(emotion_code, emotion_code)
            gender = "female" if str(row.get("gender","")).upper() == "F" else "male"
            method = row.get("method", "")  # script vs improv
            agreement = int(row.get("agreement", 0))

            # Only use high-agreement samples (all annotators agreed)
            if agreement < 3:
                continue

            # Path gives us the utterance ID
            utt_id = str(row.get("path", "")).split("/")[-1] if pd.notna(row.get("path")) else "unknown"

            # Build training text: emotional context vignette
            text = (
                f"[EMOTIONAL CONTEXT — IEMOCAP]\n"
                f"Utterance: {utt_id}\n"
                f"Speaker gender: {gender}\n"
                f"Session type: {'scripted dialogue' if method == 'script' else 'improvised scenario'}\n"
                f"Dominant emotion: {emotion_label} (annotator agreement: {agreement}/3)\n"
                f"[END CONTEXT]"
            )

            yield {
                "text": text,
                "source": "iemocap",
                "participant": utt_id,
                "labels": {
                    "phq_score": None,
                    "phq_binary": None,
                    "phq_items": None,
                    "ptsd_score": None,
                    "ptsd_binary": None,
                    "depression_severity": None,
                    "emotions": [emotion_label],
                },
                "split": "train",   # IEMOCAP has no official PHQ split; all → train
                "safety_flag": emotion_label in ["sadness", "fear", "disgust"],
            }


class RSELoader:
    """
    Loads the Rosenberg Self-Esteem Scale dataset.
    Source: archive__8_.zip → RSE/data.csv
    Columns: Q1–Q10, gender, age, source, country (TSV format)
    """

    RSE_QUESTIONS = {
        "Q1":  "I feel that I am a person of worth, at least on an equal plane with others.",
        "Q2":  "I feel that I have a number of good qualities.",
        "Q3":  "All in all, I am inclined to feel that I am a failure.",  # reverse scored
        "Q4":  "I am able to do things as well as most other people.",
        "Q5":  "I feel I do not have much to be proud of.",               # reverse scored
        "Q6":  "I take a positive attitude toward myself.",
        "Q7":  "On the whole, I am satisfied with myself.",
        "Q8":  "I wish I could have more respect for myself.",            # reverse scored
        "Q9":  "I certainly feel useless at times.",                     # reverse scored
        "Q10": "At times I think I am no good at all.",                  # reverse scored
    }
    REVERSE_ITEMS = {"Q3", "Q5", "Q8", "Q9", "Q10"}
    RESPONSE_LABELS = {1: "Strongly Agree", 2: "Agree", 3: "Disagree", 4: "Strongly Disagree"}

    def __init__(self, zip_path: str):
        self.zip_path = zip_path

    def rse_score(self, row: pd.Series) -> int:
        """Compute total RSE score (10–40, higher = higher self-esteem)."""
        total = 0
        for q in self.RSE_QUESTIONS:
            val = row.get(q)
            if pd.isna(val):
                continue
            val = int(val)
            if q in self.REVERSE_ITEMS:
                val = 5 - val   # reverse: 1→4, 2→3, 3→2, 4→1
            total += val
        return total

    def load(self) -> Iterator[Dict[str, Any]]:
        with zipfile.ZipFile(self.zip_path) as z:
            with z.open("RSE/data.csv") as f:
                df = pd.read_csv(f, sep="\t")

        # Keep a balanced sample — RSE has many rows, cap contribution
        MAX_SAMPLES = 5_000
        df = df.sample(min(len(df), MAX_SAMPLES), random_state=42)

        gender_map = {1: "male", 2: "female", 3: "other", 0: "unspecified"}

        for _, row in df.iterrows():
            score = self.rse_score(row)
            gender_code = int(row.get("gender", 0)) if pd.notna(row.get("gender")) else 0
            gender = gender_map.get(gender_code, "unspecified")
            age = int(row["age"]) if pd.notna(row.get("age")) else None
            low_esteem = score < 20

            # Build item-response text
            item_lines = []
            for q, question_text in self.RSE_QUESTIONS.items():
                val = row.get(q)
                if pd.notna(val):
                    label = self.RESPONSE_LABELS.get(int(val), str(val))
                    item_lines.append(f"  Q: {question_text}\n  A: {label}")

            text = (
                f"[SELF-ESTEEM ASSESSMENT — RSE]\n"
                f"Respondent: {gender}, age {age if age else 'unknown'}\n"
                f"Rosenberg Self-Esteem Scale Total: {score}/40\n"
                f"{'Low self-esteem indicated (score < 20).' if low_esteem else 'Normal to high self-esteem range.'}\n\n"
                f"Item responses:\n" + "\n\n".join(item_lines) + "\n[END ASSESSMENT]"
            )

            yield {
                "text": text,
                "source": "rse_scale",
                "participant": None,
                "labels": {
                    "phq_score": None,
                    "phq_binary": None,
                    "phq_items": None,
                    "ptsd_score": None,
                    "ptsd_binary": None,
                    "depression_severity": None,
                    "emotions": None,
                },
                "split": "train",
                "safety_flag": low_esteem and score < 15,   # very low esteem → safety flag
            }


class ClinicalDatasetLoader:
    """
    Orchestrator: loads all sources, applies safety filtering,
    and yields unified records ready for tokenisation.
    """

    def __init__(
        self,
        daic_woz_dir: str,
        iemocap_zip: str,
        rse_zip: str,
        safety_filter: bool = True,
    ):
        self.loaders = [
            DAICWOZLoader(daic_woz_dir),
            IEMOCAPLoader(iemocap_zip),
            RSELoader(rse_zip),
        ]
        self.safety_filter = safety_filter
        self._stats = {"total": 0, "safety_flagged": 0, "by_source": {}, "by_split": {}}

    def iter_records(self, split: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        for loader in self.loaders:
            for record in loader.load():
                self._stats["total"] += 1
                src = record["source"]
                self._stats["by_source"][src] = self._stats["by_source"].get(src, 0) + 1
                spl = record["split"]
                self._stats["by_split"][spl] = self._stats["by_split"].get(spl, 0) + 1

                if record["safety_flag"]:
                    self._stats["safety_flagged"] += 1
                    # Safety-flagged records: still usable as context/input,
                    # but should not be generation targets in loss computation.
                    record["text"] = "[SAFETY-REVIEW REQUIRED]\n" + record["text"]

                if split is None or record["split"] == split:
                    yield record

    def stats(self) -> dict:
        return self._stats


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Adjust paths to where your uploads live
    loader = ClinicalDatasetLoader(
        daic_woz_dir="/mnt/user-data/uploads",
        iemocap_zip="/mnt/user-data/uploads/archive__7___1_.zip",
        rse_zip="/mnt/user-data/uploads/archive__8_.zip",
        safety_filter=True,
    )

    print("Loading all records...\n")
    records = list(loader.iter_records())
    stats = loader.stats()

    print(f"Total records loaded : {stats['total']}")
    print(f"Safety-flagged       : {stats['safety_flagged']}")
    print(f"By source            : {stats['by_source']}")
    print(f"By split             : {stats['by_split']}")
    print()
    print("=== Sample record (DAIC-WOZ) ===")
    daic = next(r for r in records if r["source"] == "daic_woz")
    print(daic["text"])
    print("\nlabels:", daic["labels"])
    print()
    print("=== Sample record (IEMOCAP) ===")
    iem = next(r for r in records if r["source"] == "iemocap")
    print(iem["text"])
    print()
    print("=== Sample record (RSE) ===")
    rse = next(r for r in records if r["source"] == "rse_scale")
    print(rse["text"])
