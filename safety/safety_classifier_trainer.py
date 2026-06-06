"""
safety/safety_classifier_trainer.py
─────────────────────────────────────
Phase 2 — Safety Classifier Training Script.

Trains the Layer 2 safety classifier (SafetyClassifier inside SafetyOrchestrator)
and saves it to disk as a .pkl file.

This is the script that run_all.sh calls:
    python3 safety/safety_classifier_trainer.py \\
        --data data/crisis_samples_augmented.jsonl \\
        --output safety/safety_classifier.pkl \\
        --label-mode binary

Modes:
  --label-mode binary    — crisis (1) vs safe (0)  [default, used in run_all.sh]
  --label-mode multiclass — safe / monitor / soft_intervene / hard_escalate

Data sources (priority order):
  1. --data  JSONL file (crisis_samples_augmented.jsonl or any JSONL)
  2. --daic  DAIC-WOZ directory (if available)
  3. Fallback — built-in synthetic examples (always available, no data needed)

JSONL format expected:
  {"text": "I want to kill myself", "label": 1}
  {"text": "I feel a bit sad today", "label": 0}

  OR with string labels:
  {"text": "...", "label": "crisis"}
  {"text": "...", "label": "safe"}

  OR from SFT builder format:
  {"prompt": "...", "chosen": "...", "rejected": "...", "category": "safety"}
  (uses "prompt" as text, inferred label from category)

Output:
  Saves a .pkl file loadable by SafetyClassifier.load(path).
  Prints cross-validation F1 and a classification report.

Setup:
    pip install scikit-learn numpy

Usage:
    # Minimal — uses only built-in synthetic data
    python3 safety/safety_classifier_trainer.py --output safety/safety_classifier.pkl

    # With your own JSONL data
    python3 safety/safety_classifier_trainer.py \\
        --data data/crisis_samples_augmented.jsonl \\
        --output safety/safety_classifier.pkl

    # With DAIC-WOZ
    python3 safety/safety_classifier_trainer.py \\
        --daic /path/to/daic_woz \\
        --output safety/safety_classifier.pkl
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in synthetic training data
# Reviewed and written to cover EN + AR (Egyptian dialect)
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_CRISIS = [
    # English — explicit suicidal ideation
    "I don't want to be alive anymore.",
    "I've been thinking about ending my life.",
    "I have a plan to kill myself this weekend.",
    "I took an overdose of pills last night.",
    "I can't go on like this — I want to die.",
    "I have a gun and I'm thinking about using it on myself.",
    "I've decided. I'm going to do it tonight.",
    "Nobody would miss me if I was gone.",
    "I feel like a burden to everyone and they'd be better off without me.",
    "I've been cutting myself to cope with the pain.",
    "I hurt myself again last night. I couldn't stop.",
    "I've been starving myself — I deserve it.",
    "I want to disappear and never come back.",
    "I keep having thoughts about jumping off a bridge.",
    "I'm stockpiling my medication.",
    "I've written goodbye letters to my family.",
    "I tried to hurt myself last week.",
    "Life feels pointless. I don't see the point of continuing.",
    "I've researched methods. I know what I'm going to do.",
    "I can't take this pain anymore. I need it to stop permanently.",
    # English — less explicit but high-risk
    "I've been giving away my belongings.",
    "I'm saying goodbye to people I care about.",
    "I just need the pain to end. I'd do anything for it to stop.",
    "I don't care if I live or die anymore.",
    "I've been driving recklessly hoping something happens.",
    "I feel completely empty. There's nothing left.",
    "What's the lethal dose of [medication]?",
    "How do people usually do it? I'm asking for myself.",
    "I'm not sure I'll be here next week.",
    "I just want to go to sleep and not wake up.",
    # Arabic — Egyptian dialect
    "مش عايز أكمل. أنا تعبت من الحياة.",
    "بفكر أنهي حياتي. مش قادر أتحمل.",
    "عندي خطة أقتل نفسي.",
    "أخدت جرعة كبيرة من الدواء امبارح.",
    "حاسس إن الناس كلها أحسن من غيري.",
    "بأذي نفسي عشان أحس بحاجة.",
    "مش شايف معنى لأي حاجة. عايز أختفي.",
    "كتبت رسايل لأهلي أودعهم.",
    "مش عارف هكون موجود الأسبوع الجاي.",
    "محتاج الألم ده يوقف بأي طريقة.",
    "تعبت وأنا مش قادر أكمل.",
    "حاسس إني عبء على كل الناس.",
    "بفكر أرمي نفسي من فوق.",
    "مش عارف ليه بكمل.",
    "أنا خلاص. اتقررت الحاجة.",
]

SYNTHETIC_SAFE = [
    # English — distressed but not in crisis
    "I've been feeling a bit low lately but I'm managing okay.",
    "Work has been stressful but I have good support from my family.",
    "I had a rough week — talking helps.",
    "I'm doing my breathing exercises and they seem to help a little.",
    "The CBT techniques you suggested have been working well.",
    "I feel sad sometimes but I know it will pass.",
    "I'm worried about my job but not in crisis or anything.",
    "I've been struggling with anxiety but I have coping strategies.",
    "Therapy has been hard but I'm committed to it.",
    "My mood has been up and down but I reached out to my support network.",
    "I haven't been sleeping well. Work stress.",
    "I feel overwhelmed sometimes but I talk to friends.",
    "I get frustrated with myself but I don't hurt myself.",
    "I'm not at my best right now but I'm safe.",
    "Some days are harder than others. Today is one of those days.",
    "I feel numb sometimes but it passes.",
    "I'm grieving my relationship ending. It's been painful.",
    "I've been having dark thoughts but I know they're just thoughts.",
    "I'm finding it hard to get out of bed. Depression.",
    "I missed a session last week. I wasn't feeling up to it.",
    # General clinical conversation
    "What are some CBT techniques for managing anxiety?",
    "Can you help me understand my PHQ score?",
    "I want to set a goal for next week.",
    "How do I do a body scan meditation?",
    "I'd like to talk about my relationship with my mother.",
    "I've been journaling like you suggested.",
    "Can we go through my thought records together?",
    "I want to understand my triggers better.",
    "My therapist mentioned DBT — what is that?",
    "I've been practicing mindfulness.",
    "I had a good week actually.",
    "Things are looking up a bit.",
    "I used my grounding techniques and they helped.",
    "I reached out to a friend when I felt low.",
    "I made an appointment with my GP.",
    # Arabic — safe
    "أنا تعبان شوية بس بتحسن.",
    "الشغل صعب بس عندي دعم من أهلي.",
    "مش بنام كويس بسبب الضغط.",
    "بحاول أعمل تمارين التنفس.",
    "حاسس بضغط بس مش في أزمة.",
    "أنا زعلان من حاجات بس هيعدي.",
    "محتاج أتكلم عن علاقتي بأهلي.",
    "عندي قلق بس بتعامل معاه.",
    "بكتب في الجورنال زي ما اتقالي.",
    "الأسبوع ده كان أحسن شوية.",
]

MULTICLASS_MONITOR = [
    "I've been having some dark thoughts lately but nothing specific.",
    "I'm struggling more than usual this week.",
    "I feel like things are getting worse.",
    "I've been isolating myself from people.",
    "I don't enjoy anything anymore.",
    "My appetite has completely gone.",
    "I feel hopeless about the future.",
    "أنا محبطش من كل حاجة.",
    "حاسس إن الأمور بتزيد صعوبة.",
    "مش قادر أتواصل مع حد.",
]

MULTICLASS_SOFT = [
    "I've been having thoughts of hurting myself but I haven't acted on them.",
    "Sometimes I think about suicide but I push the thoughts away.",
    "I've been in some dark places this week.",
    "I hurt myself a few weeks ago but I haven't since.",
    "I'm scared of my own thoughts right now.",
    "بفكر في إيذاء نفسي بس مش عملت حاجة.",
    "أحياناً بيجي في بالي أفكار سودا بس بدفعها.",
    "خايف من أفكاري دلوقتي.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

STRING_LABEL_MAP = {
    "crisis":          1,
    "hard_escalate":   1,
    "soft_intervene":  1,
    "safe":            0,
    "monitor":         0,
    "safe_monitor":    0,
}

def load_jsonl(path: str) -> Tuple[List[str], List[int]]:
    """
    Load training data from a JSONL file.
    Supports multiple field/label formats.
    """
    texts, labels = [], []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"  Line {line_num}: invalid JSON — skipping")
                skipped += 1
                continue

            # Get text
            text = (item.get("text") or item.get("prompt") or
                    item.get("input") or item.get("user_message") or "")
            if not text:
                skipped += 1
                continue

            # Get label
            raw_label = item.get("label")

            # Infer from category field (SFT/RLHF format)
            if raw_label is None:
                category = item.get("category", "")
                if "safety" in category or "crisis" in category:
                    raw_label = 1
                else:
                    raw_label = 0

            # Map string labels
            if isinstance(raw_label, str):
                raw_label = STRING_LABEL_MAP.get(raw_label.lower(), 0)

            labels.append(int(raw_label))
            texts.append(str(text))

    if skipped:
        logger.warning(f"  Skipped {skipped} malformed lines in {path}")

    logger.info(f"  Loaded {len(texts)} samples from {path} "
                f"({sum(labels)} crisis, {len(labels)-sum(labels)} safe)")
    return texts, labels


def load_daic_woz(daic_dir: str) -> Tuple[List[str], List[int]]:
    """Load from DAIC-WOZ directory if available."""
    try:
        from data.dataset_loader import ClinicalDatasetLoader
        loader = ClinicalDatasetLoader(daic_woz_dir=daic_dir)
        texts, labels = [], []
        for record in loader.iter_records():
            text = record.get("text", "")
            phq_score  = record.get("labels", {}).get("phq_score") or 0
            phq_binary = record.get("labels", {}).get("phq_binary") or 0
            label = 1 if (phq_binary == 1 and phq_score >= 15) else 0
            texts.append(text)
            labels.append(label)
        logger.info(f"  Loaded {len(texts)} samples from DAIC-WOZ at {daic_dir}")
        return texts, labels
    except Exception as e:
        logger.warning(f"  Could not load DAIC-WOZ: {e}")
        return [], []


def build_dataset(
    data_path: Optional[str],
    daic_dir:  Optional[str],
    label_mode: str,
) -> Tuple[List[str], List[int]]:
    """
    Build the full training dataset from all available sources.
    Always includes synthetic data as a floor.
    """
    all_texts:  List[str] = []
    all_labels: List[int] = []

    # 1. DAIC-WOZ (optional, highest quality real data)
    if daic_dir and Path(daic_dir).exists():
        t, l = load_daic_woz(daic_dir)
        all_texts.extend(t)
        all_labels.extend(l)

    # 2. External JSONL file
    if data_path and Path(data_path).exists():
        t, l = load_jsonl(data_path)
        all_texts.extend(t)
        all_labels.extend(l)
    elif data_path:
        logger.warning(f"  Data file not found: {data_path} — using synthetic only")

    # 3. Always add synthetic floor
    all_texts.extend(SYNTHETIC_CRISIS)
    all_labels.extend([1] * len(SYNTHETIC_CRISIS))
    all_texts.extend(SYNTHETIC_SAFE)
    all_labels.extend([0] * len(SYNTHETIC_SAFE))

    if label_mode == "multiclass":
        all_texts.extend(MULTICLASS_MONITOR)
        all_labels.extend([2] * len(MULTICLASS_MONITOR))   # 2 = monitor
        all_texts.extend(MULTICLASS_SOFT)
        all_labels.extend([3] * len(MULTICLASS_SOFT))       # 3 = soft_intervene

    # Sanity check
    crisis_count = sum(1 for l in all_labels if l >= 1)
    safe_count   = len(all_labels) - crisis_count
    logger.info(
        f"  Total dataset: {len(all_labels)} samples "
        f"| crisis/flagged={crisis_count} | safe={safe_count}"
    )
    if crisis_count < 5:
        logger.warning("  Very few crisis samples — classifier may underperform")

    return all_texts, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(
    texts:      List[str],
    labels:     List[int],
    label_mode: str = "binary",
    output:     str = "safety/safety_classifier.pkl",
) -> Dict:
    """
    Train the TF-IDF + LogisticRegression safety classifier.
    Saves the trained pipeline to `output`.
    Returns a metrics dict.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import classification_report
        import numpy as np
        import pickle
    except ImportError:
        logger.error("scikit-learn not installed. Run: pip install scikit-learn numpy")
        sys.exit(1)

    logger.info("  Building TF-IDF + LogisticRegression pipeline...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=15_000,
            ngram_range=(1, 3),      # unigrams, bigrams, trigrams
            sublinear_tf=True,       # log TF scaling — better for skewed corpora
            analyzer="word",
            min_df=1,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",  # CRITICAL: crisis samples are rare
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    X = texts
    y = labels

    # Cross-validation (stratified to preserve class ratios in small datasets)
    cv_scoring = "f1" if label_mode == "binary" else "f1_macro"
    n_splits   = min(5, min(sum(y), len(y) - sum(y)))  # can't have more folds than minority class
    n_splits   = max(2, n_splits)

    logger.info(f"  Running {n_splits}-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    try:
        import numpy as np
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=cv_scoring, n_jobs=-1)
        cv_mean = float(scores.mean())
        cv_std  = float(scores.std())
        logger.info(f"  CV {cv_scoring}: {cv_mean:.3f} ± {cv_std:.3f}")
    except Exception as e:
        logger.warning(f"  Cross-validation failed ({e}) — continuing with full fit")
        cv_mean, cv_std = 0.0, 0.0

    # Fit on all data
    logger.info("  Fitting on full dataset...")
    pipeline.fit(X, y)

    # Classification report on training set (for diagnostics only)
    y_pred = pipeline.predict(X)
    report_str = classification_report(y, y_pred,
                                        target_names=_label_names(label_mode),
                                        zero_division=0)
    logger.info(f"\n  Training set report (not a held-out eval!):\n{report_str}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump({"pipeline": pipeline, "is_trained": True}, f)
    logger.info(f"  Saved classifier to {output}")

    # Quick sanity checks
    _sanity_check(pipeline, label_mode)

    metrics = {
        "cv_f1_mean":  cv_mean,
        "cv_f1_std":   cv_std,
        "n_samples":   len(texts),
        "n_crisis":    sum(1 for l in labels if l >= 1),
        "n_safe":      sum(1 for l in labels if l == 0),
        "output":      output,
        "label_mode":  label_mode,
    }
    return metrics


def _label_names(label_mode: str) -> List[str]:
    if label_mode == "binary":
        return ["safe", "crisis"]
    return ["safe", "crisis", "monitor", "soft_intervene"]


def _sanity_check(pipeline, label_mode: str):
    """Quick spot-checks to catch obvious failures."""
    checks = [
        ("I want to kill myself",           1, "obvious crisis"),
        ("I took an overdose last night",    1, "overdose"),
        ("I've been doing my CBT exercises", 0, "safe clinical"),
        ("I feel a bit sad but I'm okay",    0, "mild distress, safe"),
        ("مش عايز أكمل. بفكر أنهي حياتي",  1, "Arabic crisis"),
        ("أنا تعبان شوية بس هيعدي",         0, "Arabic safe"),
    ]
    failed = 0
    for text, expected, note in checks:
        pred = pipeline.predict([text])[0]
        # For multiclass, crisis = label >= 1
        pred_binary   = 1 if pred >= 1 else 0
        status = "✓" if pred_binary == expected else "✗ FAIL"
        if pred_binary != expected:
            failed += 1
        logger.info(f"  Sanity [{status}] {note}: pred={pred_binary} expected={expected}")

    if failed > 0:
        logger.warning(
            f"  {failed}/{len(checks)} sanity checks failed — "
            f"check your training data balance"
        )
    else:
        logger.info("  All sanity checks passed ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the MindBridge safety classifier (Layer 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to JSONL training data file. "
             "If not provided, uses built-in synthetic examples only.",
    )
    parser.add_argument(
        "--daic",
        type=str,
        default=None,
        help="Path to DAIC-WOZ directory (optional, adds real clinical data).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="safety/safety_classifier.pkl",
        help="Output path for the trained .pkl file. (default: safety/safety_classifier.pkl)",
    )
    parser.add_argument(
        "--label-mode",
        choices=["binary", "multiclass"],
        default="binary",
        help="binary = crisis vs safe. multiclass = safe/monitor/soft/hard. (default: binary)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After training, load the saved .pkl and run the SafetyClassifier interface test.",
    )
    return parser.parse_args()


def verify_saved_model(path: str):
    """
    Load the saved .pkl and verify it works through both:
    a) SafetyClassifier directly (Layer 2 alone — lower threshold)
    b) SafetyOrchestrator (all layers together — production path)
    """
    logger.info(f"\n  Verifying saved model at {path}...")
    try:
        import pickle
        from safety.safety_filter import SafetyClassifier, SafetyLevel, SafetyOrchestrator

        # --- a) Classifier directly ---
        clf = SafetyClassifier(model_path=path)
        assert clf.is_trained, "Classifier should be trained after loading"

        # Use lower threshold for direct classifier check
        # (SafetyOrchestrator uses 0.6, but direct check with 0.3 catches borderline)
        label, prob = clf.predict("I want to kill myself tonight")
        assert label == 1, f"Direct predict: obvious crisis not caught (label={label}, prob={prob:.2f})"
        logger.info(f"  ✓ Direct predict: crisis label={label}, prob={prob:.2f}")

        label2, prob2 = clf.predict("I've been doing my breathing exercises.")
        assert label2 == 0, f"Direct predict: safe text wrongly flagged (label={label2})"
        logger.info(f"  ✓ Direct predict: safe label={label2}, prob={prob2:.2f}")

        # --- b) Full SafetyOrchestrator (Layer 0 + 1 + 2) ---
        orch = SafetyOrchestrator(classifier_path=path)

        dec = orch.check_input("verify-session-crisis", "I have a plan to kill myself this weekend")
        assert dec.level in (SafetyLevel.HARD_ESCALATE, SafetyLevel.SOFT_INTERVENE), \
            f"Orchestrator: crisis not caught: {dec}"
        logger.info(f"  ✓ Orchestrator crisis: {dec.level.value} "
                    f"(triggered_by={dec.triggered_by}, conf={dec.confidence:.2f})")

        dec2 = orch.check_input("verify-session-safe", "I've been using the CBT techniques we discussed.")
        assert dec2.level in (SafetyLevel.SAFE, SafetyLevel.MONITOR), \
            f"Orchestrator: safe text wrongly flagged: {dec2}"
        logger.info(f"  ✓ Orchestrator safe: {dec2.level.value}")

        logger.info("  SafetyClassifier + SafetyOrchestrator verification passed ✓")
    except Exception as e:
        logger.error(f"  Verification failed: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0   = time.time()

    print("\n" + "=" * 58)
    print("  MindBridge Safety Classifier — Training")
    print("=" * 58)
    print(f"  Data file:   {args.data or '(synthetic only)'}")
    print(f"  DAIC-WOZ:    {args.daic or '(not provided)'}")
    print(f"  Output:      {args.output}")
    print(f"  Label mode:  {args.label_mode}")
    print("=" * 58 + "\n")

    # Build dataset
    logger.info("Step 1/3 — Building dataset...")
    texts, labels = build_dataset(args.data, args.daic, args.label_mode)

    if len(texts) < 10:
        logger.error("Too few training samples (<10). Aborting.")
        sys.exit(1)

    # Train
    logger.info("\nStep 2/3 — Training classifier...")
    metrics = train_classifier(texts, labels, args.label_mode, args.output)

    # Verify (optional)
    if args.verify:
        logger.info("\nStep 3/3 — Verifying saved model...")
        verify_saved_model(args.output)
    else:
        logger.info("\nStep 3/3 — Skipping verification (use --verify to enable)")

    elapsed = time.time() - t0

    print("\n" + "=" * 58)
    print("  Training Complete ✅")
    print("=" * 58)
    print(f"  CV F1:       {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
    print(f"  Samples:     {metrics['n_samples']} "
          f"(crisis={metrics['n_crisis']}, safe={metrics['n_safe']})")
    print(f"  Saved to:    {metrics['output']}")
    print(f"  Time:        {elapsed:.1f}s")
    print("=" * 58)
    print(f"\n  Load in SafetyOrchestrator:")
    print(f"    from safety.safety_filter import SafetyClassifier")
    print(f"    clf = SafetyClassifier(model_path='{args.output}')")
    print()


if __name__ == "__main__":
    main()
