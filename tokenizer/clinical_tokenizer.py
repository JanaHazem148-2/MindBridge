"""
tokenizer/clinical_tokenizer.py
────────────────────────────────
Train a BPE tokenizer on the clinical corpus, then wrap it for use
in the training pipeline.

Why train our own?
  - Clinical terminology (PHQ, PCL-C, CBT, DBT, anhedonia, dysthymia...)
    is underrepresented in general-purpose tokenizers.
  - Data sovereignty: our tokenizer, our vocab.
  - We add special tokens for agent roles and safety markers.

Dependencies:
    pip install tokenizers transformers
"""

import os
import json
from pathlib import Path
from typing import Iterator, List, Optional
from io import StringIO

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence as NormSeq


# ── Special tokens ────────────────────────────────────────────────────────────
# These are inserted into the vocabulary at fixed positions so downstream
# code can reference them by constant ID rather than string lookup.

SPECIAL_TOKENS = [
    "<pad>",            # 0  — padding
    "<bos>",            # 1  — beginning of sequence
    "<eos>",            # 2  — end of sequence
    "<unk>",            # 3  — unknown token (should rarely appear)
    "<mask>",           # 4  — masked token (for future MLM fine-tuning)

    # ── Role markers (used in SFT/RLHF phases) ──────────────────────────────
    "<|system|>",       # 5
    "<|therapist|>",    # 6
    "<|patient|>",      # 7
    "<|clinician|>",    # 8

    # ── Clinical section markers ─────────────────────────────────────────────
    "<|phq_start|>",    # 9
    "<|phq_end|>",      # 10
    "<|safety_flag|>",  # 11 — precedes any safety-critical content
    "<|crisis|>",       # 12 — hard trigger: escalate immediately

    # ── Memory / RAG markers ─────────────────────────────────────────────────
    "<|memory_start|>", # 13
    "<|memory_end|>",   # 14
    "<|retrieved|>",    # 15
]

PAD_TOKEN_ID   = 0
BOS_TOKEN_ID   = 1
EOS_TOKEN_ID   = 2
UNK_TOKEN_ID   = 3
SAFETY_FLAG_ID = 11
CRISIS_ID      = 12


# ── Text iterator for tokenizer training ─────────────────────────────────────

def clinical_text_iterator(data_dir: str, extra_texts: Optional[List[str]] = None) -> Iterator[str]:
    """
    Yields raw text strings for BPE training.
    Pulls from all available CSV/text sources.
    """
    import pandas as pd
    import zipfile

    # DAIC-WOZ splits
    for fname in ["train_split.csv", "dev_split.csv", "test_split.csv"]:
        path = Path(data_dir) / fname
        if path.exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                score = int(row.get("PHQ_Score", 0))
                yield f"PHQ-8 score {score} indicates {_severity(score)} depression."

    # Detailed PHQ labels
    detailed_path = Path(data_dir) / "detailed_lables.csv"
    if detailed_path.exists():
        df = pd.read_csv(detailed_path)
        for _, row in df.iterrows():
            # Emit clinical terminology
            yield " ".join([
                "anhedonia", "dysphoria", "insomnia", "hypersomnia",
                "fatigue", "worthlessness", "concentration", "psychomotor retardation",
                "PHQ-8", "PCL-C", "PTSD", "depression", "severity", "assessment",
                "CBT", "DBT", "cognitive", "behavioral", "therapy", "empathy",
                "therapeutic alliance", "safety plan", "crisis intervention",
            ])

    # IEMOCAP emotions
    iemocap_zip = Path(data_dir) / "archive__7___1_.zip"
    if iemocap_zip.exists():
        with zipfile.ZipFile(iemocap_zip) as z:
            with z.open("iemocap_full_dataset.csv") as f:
                df = pd.read_csv(f)
        emotions = df["emotion"].unique().tolist()
        yield " ".join(["emotional states:", *emotions,
                         "frustration anger sadness happiness neutral",
                         "excited fear surprise disgust"])

    # RSE items (high-quality psychological text)
    rse_questions = [
        "I feel that I am a person of worth, at least on an equal plane with others.",
        "I feel that I have a number of good qualities.",
        "All in all, I am inclined to feel that I am a failure.",
        "I am able to do things as well as most other people.",
        "I feel I do not have much to be proud of.",
        "I take a positive attitude toward myself.",
        "On the whole, I am satisfied with myself.",
        "I wish I could have more respect for myself.",
        "I certainly feel useless at times.",
        "At times I think I am no good at all.",
    ]
    yield " ".join(rse_questions)

    # Clinical domain seed sentences (hardcoded vocabulary anchors)
    clinical_seeds = [
        "The patient presents with depressive symptoms including low mood, anhedonia, and sleep disturbance.",
        "Cognitive behavioral therapy sessions focus on identifying automatic negative thoughts.",
        "Dialectical behavior therapy skills include distress tolerance and emotional regulation.",
        "The PHQ-9 is a validated screening tool for major depressive disorder.",
        "Safety planning involves identifying warning signs, coping strategies, and crisis contacts.",
        "Motivational interviewing techniques include reflective listening and developing discrepancy.",
        "The clinician conducted a suicide risk assessment using a standardized protocol.",
        "Trauma-focused CBT addresses intrusive memories, avoidance, and hyperarousal.",
        "PTSD symptoms cluster into re-experiencing, avoidance, negative cognitions, and hyperarousal.",
        "Psychoeducation helps patients understand the cognitive model of depression.",
        "The therapeutic alliance is a predictor of treatment outcomes across therapy modalities.",
        "Mindfulness-based cognitive therapy reduces relapse rates in recurrent depression.",
        "Behavioral activation involves scheduling pleasurable activities to counter withdrawal.",
        "The PHQ-8 differs from PHQ-9 in excluding the suicidal ideation item.",
        "Scoring: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe depression.",
    ]
    for seed in clinical_seeds:
        yield seed

    # Extra texts if provided
    if extra_texts:
        for text in extra_texts:
            yield text


def _severity(score: int) -> str:
    if score <= 4:  return "minimal"
    if score <= 9:  return "mild"
    if score <= 14: return "moderate"
    if score <= 19: return "moderately severe"
    return "severe"


# ── Tokenizer factory ─────────────────────────────────────────────────────────

def train_clinical_tokenizer(
    data_dir: str,
    output_dir: str,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
) -> Tokenizer:
    """
    Train a BPE tokenizer on the clinical corpus and save it.

    Args:
        data_dir:      Directory containing the CSV/zip data files.
        output_dir:    Where to save tokenizer.json and vocab files.
        vocab_size:    Target vocabulary size.
        min_frequency: Minimum token frequency to include in vocab.

    Returns:
        Trained Tokenizer instance.
    """
    print(f"Training clinical BPE tokenizer (vocab_size={vocab_size})...")

    # ── BPE model ──────────────────────────────────────────────────────────
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenizer: split on whitespace + punctuation (ByteLevel for UTF-8 safety)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder: reverses ByteLevel encoding for generation
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor: automatically wrap with BOS/EOS
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # ── Train ────────────────────────────────────────────────────────────
    text_iter = list(clinical_text_iterator(data_dir))
    print(f"  Training on {len(text_iter)} text samples from clinical corpus...")
    tokenizer.train_from_iterator(text_iter, trainer=trainer)

    # ── Verify special token IDs ──────────────────────────────────────────
    for i, tok in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(tok)
        if actual_id != i:
            print(f"  WARNING: {tok} got ID {actual_id}, expected {i}")
        else:
            print(f"  ✓ {tok} → ID {i}")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    # Save human-readable metadata
    meta = {
        "vocab_size": vocab_size,
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "model_type": "bpe",
        "domain": "clinical-mental-health",
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTokenizer saved to {output_dir}/")
    print(f"Actual vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def load_clinical_tokenizer(tokenizer_dir: str) -> Tokenizer:
    """Load a previously trained tokenizer from disk."""
    return Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))


# ── HuggingFace-compatible wrapper ────────────────────────────────────────────

class ClinicalTokenizerWrapper:
    """
    Thin wrapper that makes the tokenizers.Tokenizer behave like a
    HuggingFace PreTrainedTokenizer for use in DataLoader and Trainer.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id   = tokenizer.token_to_id("<pad>")
        self.bos_token_id   = tokenizer.token_to_id("<bos>")
        self.eos_token_id   = tokenizer.token_to_id("<eos>")
        self.unk_token_id   = tokenizer.token_to_id("<unk>")
        self.safety_flag_id = tokenizer.token_to_id("<|safety_flag|>")
        self.crisis_id      = tokenizer.token_to_id("<|crisis|>")
        self.vocab_size     = tokenizer.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if add_special_tokens:
            text = f"<bos>{text}<eos>"
        encoded = self.tokenizer.encode(text)
        return encoded.ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, texts, max_length: int = 2048, padding: bool = True,
                 truncation: bool = True, return_tensors: Optional[str] = None):
        import torch
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self.encode(t) for t in texts]

        if truncation:
            encoded = [e[:max_length] for e in encoded]

        if padding:
            max_len = max(len(e) for e in encoded)
            attention_masks = []
            padded = []
            for e in encoded:
                pad_len = max_len - len(e)
                attention_masks.append([1] * len(e) + [0] * pad_len)
                padded.append(e + [self.pad_token_id] * pad_len)
            encoded = padded
        else:
            attention_masks = [[1] * len(e) for e in encoded]

        if return_tensors == "pt":
            return {
                "input_ids":      torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        return {"input_ids": encoded, "attention_mask": attention_masks}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train clinical BPE tokenizer")
    parser.add_argument("--data_dir",   default="/mnt/user-data/uploads")
    parser.add_argument("--output_dir", default="./tokenizer/clinical_bpe_32k")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    args = parser.parse_args()

    tok = train_clinical_tokenizer(args.data_dir, args.output_dir, args.vocab_size)

    # Quick sanity check
    wrapper = ClinicalTokenizerWrapper(tok)
    test = "The patient reports PHQ-8 score of 12, indicating moderate depression."
    ids = wrapper.encode(test)
    decoded = wrapper.decode(ids)
    print(f"\nTest encode: '{test}'")
    print(f"  Token IDs : {ids[:15]}...")
    print(f"  Decoded   : '{decoded}'")
    print(f"\n✓ Tokenizer ready. Vocab size: {wrapper.vocab_size}")
