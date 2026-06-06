"""
configs/training_config.py
──────────────────────────
All hyperparameters for the pre-training run.
Separating these from model architecture makes sweeps & reproducibility clean.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class TrainingConfig:
    # ── Run identity ─────────────────────────────────────────────────────────
    run_name: str = "mindbridge-pretrain-v1"
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    seed: int = 42

    # ── Data ─────────────────────────────────────────────────────────────────
    train_data_dir: str = "./data/tokenized/train"
    eval_data_dir: str = "./data/tokenized/eval"
    data_sources: List[str] = field(default_factory=lambda: [
        "daic_woz",          # Depression/PTSD clinical interviews
        "iemocap",           # Emotion-labeled speech transcripts
        "rse_scale",         # Rosenberg self-esteem assessment
        "synthetic_cbt",     # CBT/DBT synthetic dialogues (generated separately)
        "pubmed_abstracts",  # General biomedical (downloaded separately)
    ])
    # Sampling weights per source (must sum to ~1.0)
    data_weights: List[float] = field(default_factory=lambda: [
        0.35,   # DAIC-WOZ — highest weight: most clinically relevant
        0.15,   # IEMOCAP — emotional grounding
        0.05,   # RSE — self-assessment patterns
        0.30,   # Synthetic CBT/DBT dialogues
        0.15,   # PubMed — general medical context
    ])

    # ── Batch & sequence ─────────────────────────────────────────────────────
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    # Effective batch size = 4 × 8 GPUs × 16 grad_accum = 512 sequences
    # At 2048 tokens each → ~1M tokens / gradient step

    max_seq_len: int = 2048

    # ── Optimisation ─────────────────────────────────────────────────────────
    learning_rate: float = 3e-4           # Peak LR with cosine decay
    min_learning_rate: float = 3e-5       # Floor: 10% of peak
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0            # Gradient clipping

    # ── Schedule ─────────────────────────────────────────────────────────────
    lr_scheduler_type: str = "cosine"     # cosine | linear | constant
    warmup_steps: int = 2000
    # Training budget: aim for ~100B tokens processed total
    # At 1M tokens/step this is ~100K steps
    max_steps: int = 100_000
    num_train_epochs: int = -1            # -1 means use max_steps

    # ── Mixed precision ───────────────────────────────────────────────────────
    bf16: bool = True                     # A100 supports bf16 natively
    fp16: bool = False                    # Don't mix bf16 and fp16
    tf32: bool = True                     # A100 tensor cores — free speedup

    # ── FSDP (multi-GPU) ─────────────────────────────────────────────────────
    fsdp: bool = True
    fsdp_sharding_strategy: str = "FULL_SHARD"   # ZeRO-3 equivalent
    fsdp_auto_wrap_policy: str = "TRANSFORMER_BASED_WRAP"
    fsdp_cpu_offload: bool = False        # Only enable if OOM on GPU

    # ── Gradient checkpointing ────────────────────────────────────────────────
    gradient_checkpointing: bool = True   # Saves ~30% memory, ~20% slower

    # ── Evaluation & saving ───────────────────────────────────────────────────
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 5             # Keep last 5 checkpoints
    logging_steps: int = 10

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer_path: str = "./tokenizer/clinical_bpe_32k"
    tokenizer_vocab_size: int = 32_000

    # ── Clinical-specific ────────────────────────────────────────────────────
    # Safety: block any sample where PHQ_Binary==1 from being used as
    # positive generation targets during pre-training (use as context only)
    safety_filter_crisis_samples: bool = True

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def effective_batch_size(self) -> int:
        n_gpus = int(os.environ.get("WORLD_SIZE", 1))
        return (
            self.per_device_train_batch_size
            * n_gpus
            * self.gradient_accumulation_steps
        )

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.max_seq_len
