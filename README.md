# MindBridge — Phase 1-3 Complete

Bilingual Arabic/English AI mental health companion, clinically supervised.

## Status After Gap Fill

| Component | Status | Details |
|-----------|--------|---------|
| transformer.py | Done | 0.16B / 1.3B / 7B configs |
| pretrain.py | Done | FSDP multi-GPU, loss 139→13 |
| clinical_tokenizer.py | Done | BPE + medical special tokens |
| safety/safety_filter.py | Done | 3-layer pipeline, 4 severity levels |
| safety/arabic_patterns.py | **NEW** | Egyptian + MSA Arabic crisis patterns |
| sft/sft_trainer.py | Done | LoRA 48-layer, loss→0.25 |
| sft_train.jsonl | **UPDATED** | +150 Arabic pairs (total ~3387) |
| rlhf/dpo_pairs_full.py | **NEW** | 50 pairs, 5 clinical categories |
| rlhf/clinician_scorer.py | **NEW** | Auto + LLM-judge + human annotation |
| data/crisis_samples_augmented.jsonl | **NEW** | 200+ EN+AR crisis samples |
| scripts/build_complete.py | **NEW** | Master build/status script |

## Quick Start

```bash
pip install -r requirements.txt
pip install chromadb sentence-transformers   # for RAG

python scripts/build_complete.py --all       # build everything
python scripts/build_complete.py --status    # check what's built
```

## What Was Added

### Arabic Safety (`safety/arabic_patterns.py`)
- Hard triggers: suicidal ideation MSA + Egyptian (`عايز أنهي حياتي`, `هانتحر`)
- Hard triggers: self-harm (`بجرح نفسي`), abuse disclosures (`اتاذيت`)
- Soft triggers: hopelessness, passive ideation (`ياريت ما اصحيتش`)
- Monitor: distress, sleep, anxiety patterns
- Text normalisation: diacritics, alef variants, teh marbuta
- Resources: Egypt 08008880700, Saudi 1919, UAE 800HOPE

### Arabic in SafetyOrchestrator (`safety/safety_filter.py`)
Arabic filter runs as Layer 1b between regex and ML classifier.
Auto-detects Arabic script. Arabic-specific escalation messages.

### Crisis Samples 200+ (`data/crisis_samples_augmented.jsonl`)
EN hard x3 + EN soft x2 + AR hard x2 + AR soft + safe = 200+

### DPO Data + Scorer (`rlhf/`)
```bash
# Auto-score all pairs
python rlhf/clinician_scorer.py --score --input rlhf/dpo_pairs_full.jsonl

# Human annotation session
python rlhf/clinician_scorer.py --annotate --n 50 --annotator dr_ahmed

# Inter-rater reliability
python rlhf/clinician_scorer.py --iir --file-a ann_a.jsonl --file-b ann_b.jsonl
```

Clinician Scoring Rubric (0-5 per dimension):
- empathy: 0=cold/dismissive → 5=warm/validating
- safety: 0=harmful → 5=proactively safe + resources
- usefulness: 0=wrong → 5=accurate + actionable

## Training Pipeline

```
pretrain.py  →  sft_trainer.py  →  dpo_trainer.py
              (loss→0.25)         (--pairs dpo_pairs_scored.jsonl)
```

## Safety Architecture

```
User input
  ├── Layer 1a: English regex      (0ms)
  ├── Layer 1b: Arabic regex       (0ms)   ← NEW
  ├── Layer 2:  sklearn classifier (20ms)
  └── Layer 3:  LLM-as-judge      (200ms)
        ↓
SAFE / MONITOR / SOFT_INTERVENE / HARD_ESCALATE
        ↓
HARD_ESCALATE → one-way door → bypass LLM → crisis resources
```
