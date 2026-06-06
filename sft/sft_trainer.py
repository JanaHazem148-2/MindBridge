"""
sft/sft_trainer.py
───────────────────
Phase 3b: Supervised Fine-Tuning of MindBridgeLLM.

Uses LoRA (Low-Rank Adaptation) to fine-tune efficiently:
  - Freeze 99% of base model weights
  - Train only LoRA adapter matrices (rank-16, ~20M params)
  - 10–50× faster than full fine-tuning
  - Adapter can be saved/loaded separately from base model

Training objective:
  Standard next-token prediction, BUT only on assistant tokens.
  User messages and system prompt are masked from the loss.
  This prevents the model from learning to predict user turns.

Usage:
    python sft/sft_trainer.py \
        --base_checkpoint checkpoints/mindbridge-1b/step_010000 \
        --sft_data sft/sft_train.jsonl \
        --output_dir checkpoints/sft-lora-v1 \
        --epochs 3
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.model_config import ModelConfig
from model.transformer import MindBridgeLLM
from sft.sft_data_builder import format_for_training

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")


# ── LoRA Implementation ───────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adaptation.

    Instead of updating W (frozen), we learn:
        W_adapted = W + (B @ A) × (alpha / rank)

    A is initialised with random normal, B with zeros.
    At init, B @ A = 0, so the adapted model starts identical to base.

    rank:  8–64 (higher = more expressive, more params)
    alpha: scaling factor, typically = rank (so scale = 1.0)
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original  = original_linear
        self.rank      = rank
        self.scale     = alpha / rank

        in_features  = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices — small compared to original weight
        self.lora_A = nn.Linear(in_features,  rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        # Init: A~N(0,σ²), B=0 → adapter starts at zero
        nn.init.normal_(self.lora_A.weight, std=1.0 / math.sqrt(rank))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze base weight
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA into base weight for inference (removes latency overhead).
        Returns a plain nn.Linear with the adapted weights.
        """
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        merged.weight.data = (
            self.original.weight.data
            + (self.lora_B.weight @ self.lora_A.weight) * self.scale
        )
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data.clone()
        return merged


def inject_lora(
    model: MindBridgeLLM,
    rank: int = 16,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
) -> MindBridgeLLM:
    """
    Replace target linear layers with LoRA-wrapped versions.
    Freezes all other parameters.

    target_modules: which projection names to adapt.
    Default: Q, V projections in attention (standard LoRA recipe).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]  # Standard LoRA targets

    # First: freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Then: inject LoRA into target layers
    n_lora = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Navigate to parent
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], LoRALinear(module, rank=rank, alpha=alpha))
                n_lora += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA injected: {n_lora} layers | "
                f"trainable={trainable:,} ({100*trainable/total:.2f}% of total)")
    return model


def save_lora_adapter(model: MindBridgeLLM, path: str):
    """Save only the LoRA adapter weights (not the full model)."""
    lora_state = {
        name: param
        for name, param in model.state_dict().items()
        if "lora_" in name
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(lora_state, path)
    size_mb = sum(p.numel() * 4 for p in lora_state.values()) / 1e6
    logger.info(f"LoRA adapter saved: {len(lora_state)} tensors, ~{size_mb:.1f}MB → {path}")


def load_lora_adapter(model: MindBridgeLLM, path: str) -> MindBridgeLLM:
    """Load LoRA adapter weights into an already-injected model."""
    lora_state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    logger.info(f"LoRA adapter loaded: {len(lora_state)} tensors")
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}")
    return model


# ── SFT Dataset ───────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """
    Loads SFT examples from a JSONL file and tokenizes them.
    Builds a loss mask that is 1 only on assistant tokens.
    """

    # Special token strings used in format_for_training()
    THERAPIST_TOKEN = "<|therapist|>"
    PATIENT_TOKEN   = "<|patient|>"
    SYSTEM_TOKEN    = "<|system|>"
    EOS_TOKEN       = "<eos>"

    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 2048):
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        text = format_for_training(example)

        # Tokenize
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:self.max_seq_len]
        ids = torch.tensor(ids, dtype=torch.long)

        # Build loss mask: 1 on assistant tokens, 0 elsewhere
        # Strategy: find the last <|therapist|> token, mask everything before it
        therapist_id = self.tokenizer.tokenizer.token_to_id(self.THERAPIST_TOKEN)
        loss_mask = torch.zeros_like(ids)

        if therapist_id is not None:
            # Find all positions of therapist token
            therapist_positions = (ids == therapist_id).nonzero(as_tuple=True)[0]
            if len(therapist_positions) > 0:
                # Last occurrence = start of assistant response
                last_therapist = therapist_positions[-1].item()
                loss_mask[last_therapist + 1:] = 1
        else:
            # Fallback: apply loss to full sequence
            loss_mask[:] = 1

        return {
            "input_ids":  ids,
            "labels":     ids.clone(),
            "loss_mask":  loss_mask,
        }


def sft_collate(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    B = len(batch)

    input_ids   = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels      = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn_mask   = torch.zeros(B, max_len, dtype=torch.long)
    loss_mask   = torch.zeros(B, max_len, dtype=torch.float)

    for i, item in enumerate(batch):
        L = item["input_ids"].shape[0]
        input_ids[i, :L] = item["input_ids"]
        labels[i, :L]    = item["labels"]
        attn_mask[i, :L] = 1
        loss_mask[i, :L] = item["loss_mask"].float()

    return {
        "input_ids":      input_ids,
        "labels":         labels,
        "attention_mask": attn_mask,
        "loss_mask":      loss_mask,
    }


# ── SFT Trainer ───────────────────────────────────────────────────────────────

class SFTTrainer:
    """
    Fine-tunes MindBridgeLLM with LoRA using the SFT dataset.
    """

    def __init__(
        self,
        base_model: MindBridgeLLM,
        model_config: ModelConfig,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lr: float = 2e-4,
        device: Optional[torch.device] = None,
    ):
        self.config = model_config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inject LoRA
        self.model = inject_lora(
            base_model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        ).to(self.device)

        # Only optimise LoRA params
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )

    def compute_sft_loss(
        self,
        logits: torch.Tensor,      # [B, S, V]
        labels: torch.Tensor,      # [B, S]
        loss_mask: torch.Tensor,   # [B, S] — 1 on assistant tokens
    ) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask   = loss_mask[:, 1:].contiguous()

        loss = nn.CrossEntropyLoss(
            ignore_index=self.config.pad_token_id,
            reduction="none",
        )(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Only count loss on assistant tokens
        loss = loss * shift_mask.view(-1)
        n_tokens = shift_mask.sum().clamp(min=1.0)
        return loss.sum() / n_tokens

    def train(
        self,
        train_dataset: SFTDataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        grad_accum: int = 8,
        eval_dataset: Optional[SFTDataset] = None,
        save_dir: str = "checkpoints/sft-lora",
        logging_steps: int = 10,
    ):
        collate = lambda b: sft_collate(b, self.config.pad_token_id)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate)

        total_steps = len(train_loader) * num_epochs // grad_accum
        logger.info(f"SFT Training: {num_epochs} epochs, {total_steps} steps")

        self.model.train()
        global_step = 0
        self.optimizer.zero_grad()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    logits, _ = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss = self.compute_sft_loss(logits, batch["labels"], batch["loss_mask"])

                loss = loss / grad_accum
                loss.backward()
                epoch_loss += loss.item()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % logging_steps == 0:
                        avg_loss = epoch_loss * grad_accum / (step + 1)
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs} | "
                            f"Step {global_step} | "
                            f"Loss {avg_loss:.4f} | "
                            f"PPL {math.exp(min(avg_loss,20)):.2f}"
                        )

            # Save adapter after each epoch
            save_lora_adapter(self.model, os.path.join(save_dir, f"lora_epoch_{epoch+1}.pt"))

        save_lora_adapter(self.model, os.path.join(save_dir, "lora_final.pt"))
        logger.info(f"SFT complete. Adapter saved → {save_dir}")
        return self.model


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--sft_data",        default="sft/sft_train.jsonl")
    parser.add_argument("--tokenizer_dir",   default="tokenizer/clinical_bpe_32k")
    parser.add_argument("--output_dir",      default="checkpoints/sft-lora-v1")
    parser.add_argument("--epochs",          type=int, default=3)
    parser.add_argument("--batch_size",      type=int, default=4)
    parser.add_argument("--lora_rank",       type=int, default=16)
    parser.add_argument("--lr",              type=float, default=2e-4)
    args = parser.parse_args()

    # Load base model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(args.base_checkpoint, "checkpoint.pt"), map_location=device)
    model_cfg = ModelConfig(**ckpt["model_config"])
    base_model = MindBridgeLLM(model_cfg)
    base_model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded base model: {model_cfg.model_name}")

    # Load tokenizer
    from tokenizer.clinical_tokenizer import load_clinical_tokenizer, ClinicalTokenizerWrapper
    tokenizer = ClinicalTokenizerWrapper(load_clinical_tokenizer(args.tokenizer_dir))

    # Build dataset
    train_ds = SFTDataset(args.sft_data, tokenizer)
    logger.info(f"SFT dataset: {len(train_ds)} examples")

    # Train
    trainer = SFTTrainer(
        base_model=base_model,
        model_config=model_cfg,
        lora_rank=args.lora_rank,
        lr=args.lr,
        device=device,
    )
    trainer.train(
        train_dataset=train_ds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.output_dir,
    )
