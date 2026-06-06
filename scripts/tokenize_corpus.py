"""
scripts/tokenize_corpus.py
──────────────────────────
Offline pre-processing: tokenize the clinical corpus into sharded .pt files
that the training DataLoader streams efficiently.

This runs ONCE before pre-training and produces files like:
    data/tokenized/train/train_000.pt
    data/tokenized/train/train_001.pt
    ...
    data/tokenized/eval/eval_000.pt

Each .pt file is a list of 1D torch.LongTensor (token IDs).

Usage:
    python scripts/tokenize_corpus.py \
        --data_dir /mnt/user-data/uploads \
        --tokenizer_dir ./tokenizer/clinical_bpe_32k \
        --output_dir ./data/tokenized \
        --shard_size 100000
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Iterator

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import ClinicalDatasetLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


def load_tokenizer(tokenizer_dir: str):
    """Load the clinical BPE tokenizer."""
    try:
        from tokenizer.clinical_tokenizer import load_clinical_tokenizer, ClinicalTokenizerWrapper
        tok = load_clinical_tokenizer(tokenizer_dir)
        return ClinicalTokenizerWrapper(tok)
    except Exception as e:
        logger.warning(f"Could not load BPE tokenizer ({e}), using char-level fallback")
        return None


class CharLevelFallback:
    """Dev-only fallback when real tokenizer isn't available."""
    vocab_size = 32_000
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [1]  # BOS
        ids += [min(ord(c), 31999) for c in text]
        ids += [2]  # EOS
        return ids


def tokenize_records(
    tokenizer,
    records: Iterator[dict],
    max_seq_len: int = 2048,
) -> Iterator[torch.Tensor]:
    """Tokenize each record, yield 1D token ID tensors."""
    for record in records:
        text = record["text"]
        try:
            ids = tokenizer.encode(text, add_special_tokens=True)
        except Exception:
            continue

        # Truncate to max_seq_len
        ids = ids[:max_seq_len]
        if len(ids) < 4:   # too short to be useful
            continue
        yield torch.tensor(ids, dtype=torch.long)


def shard_and_save(
    token_tensors: List[torch.Tensor],
    output_dir: str,
    split: str,
    shard_idx: int,
):
    """Save a shard of token tensors to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / f"{split}_{shard_idx:03d}.pt"
    torch.save(token_tensors, path)
    total_tokens = sum(t.numel() for t in token_tensors)
    logger.info(f"Saved {path.name}: {len(token_tensors):,} seqs, {total_tokens:,} tokens")
    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       default="/mnt/user-data/uploads")
    parser.add_argument("--tokenizer_dir",  default="./tokenizer/clinical_bpe_32k")
    parser.add_argument("--output_dir",     default="./data/tokenized")
    parser.add_argument("--shard_size",     type=int, default=100_000,
                        help="Number of sequences per shard file")
    parser.add_argument("--max_seq_len",    type=int, default=2048)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_dir) or CharLevelFallback()
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load all clinical records
    loader = ClinicalDatasetLoader(
        daic_woz_dir=args.data_dir,
        iemocap_zip=os.path.join(args.data_dir, "archive__7___1_.zip"),
        rse_zip=os.path.join(args.data_dir, "archive__8_.zip"),
        safety_filter=True,
    )

    stats = {"total_tokens": 0, "total_seqs": 0, "shards": 0}

    for split in ["train", "dev", "test"]:
        logger.info(f"\nProcessing split: {split}")
        out_dir = os.path.join(args.output_dir, "train" if split == "train" else "eval")

        records = loader.iter_records(split=split)
        token_iter = tokenize_records(tokenizer, records, args.max_seq_len)

        shard_idx = 0
        buffer = []

        for tensor in token_iter:
            buffer.append(tensor)
            if len(buffer) >= args.shard_size:
                total = shard_and_save(buffer, out_dir, split, shard_idx)
                stats["total_tokens"] += total
                stats["total_seqs"]   += len(buffer)
                stats["shards"]       += 1
                shard_idx += 1
                buffer = []

        # Save remaining
        if buffer:
            total = shard_and_save(buffer, out_dir, split, shard_idx)
            stats["total_tokens"] += total
            stats["total_seqs"]   += len(buffer)
            stats["shards"]       += 1

    logger.info(f"\n{'='*50}")
    logger.info(f"Tokenization complete!")
    logger.info(f"  Total sequences : {stats['total_seqs']:,}")
    logger.info(f"  Total tokens    : {stats['total_tokens']:,}")
    logger.info(f"  Total shards    : {stats['shards']}")
    logger.info(f"  Output          : {args.output_dir}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
