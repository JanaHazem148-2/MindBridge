"""
training/trainer.py
───────────────────
Pre-training loop for MindBridgeLLM.

Key features
────────────
1. FSDP with full sharding (ZeRO-3 equivalent) for multi-GPU runs
2. Cosine LR schedule with linear warmup
3. Gradient checkpointing to reduce memory usage
4. Safety mask: crisis tokens excluded from loss computation
5. WandB-compatible logging (falls back to stdout if WandB unavailable)
6. Checkpointing with FSDP-aware save/load
7. Token-budget tracking (know how many tokens you've consumed)

Usage (single-node 8× A100):
    torchrun --nproc_per_node=8 scripts/pretrain.py --config 1b

Usage (single GPU dev run):
    python scripts/pretrain.py --config tiny --no_fsdp
"""

import os
import math
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.cuda.amp import GradScaler

# Graceful imports for optional distributed deps
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        BackwardPrefetch,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# local
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from model.transformer import MindBridgeLLM, TransformerBlock

logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenizedDataset(IterableDataset):
    """
    Streams pre-tokenized shards (`.pt` files) from a directory.
    Each shard is a list of token ID tensors.
    Supports multi-worker DataLoader.

    Pre-tokenization is done offline via scripts/tokenize_corpus.py.
    """

    def __init__(self, data_dir: str, max_seq_len: int, split: str = "train"):
        self.data_dir   = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.split       = split
        self.shards      = sorted(self.data_dir.glob(f"{split}_*.pt"))
        if not self.shards:
            # Fallback: load from CSV for development (slow but works without pre-tokenization)
            self._use_csv_fallback = True
        else:
            self._use_csv_fallback = False

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if self._use_csv_fallback:
            yield from self._csv_fallback_iter()
            return

        # Distribute shards across workers
        shards = self.shards
        if worker_info is not None:
            shards = [s for i, s in enumerate(shards) if i % worker_info.num_workers == worker_info.id]

        for shard_path in shards:
            data = torch.load(shard_path, map_location="cpu")  # list of token tensors
            # Pack tokens into fixed-length chunks
            buffer = []
            for tokens in data:
                buffer.extend(tokens.tolist())
                while len(buffer) >= self.max_seq_len:
                    chunk = buffer[:self.max_seq_len]
                    buffer = buffer[self.max_seq_len:]
                    ids    = torch.tensor(chunk, dtype=torch.long)
                    # safety_mask: 1 where token is safety-flag ID (11) or crisis ID (12)
                    safety = ((ids == 11) | (ids == 12)).long()
                    yield {"input_ids": ids, "labels": ids.clone(), "safety_mask": safety}

    def _csv_fallback_iter(self):
        """
        Development fallback: loads records directly from CSV files and
        applies a naive whitespace tokenization → integer IDs.
        Replace with proper tokenizer in production.
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.dataset_loader import ClinicalDatasetLoader

        loader = ClinicalDatasetLoader(
            daic_woz_dir="/mnt/user-data/uploads",
            iemocap_zip="/mnt/user-data/uploads/archive__7___1_.zip",
            rse_zip="/mnt/user-data/uploads/archive__8_.zip",
        )
        for record in loader.iter_records(split=self.split if self.split != "eval" else "dev"):
            # Naive character-level encoding for dev (replace with BPE in production)
            text = record["text"]
            ids  = [min(ord(c), 31999) for c in text[:self.max_seq_len]]
            ids  = torch.tensor(ids, dtype=torch.long)
            safety_flag = 1 if record["safety_flag"] else 0
            safety = torch.full_like(ids, safety_flag)
            yield {"input_ids": ids, "labels": ids.clone(), "safety_mask": safety}


def collate_fn(batch, pad_token_id: int = 0, max_seq_len: int = 2048):
    """Pad a batch of variable-length sequences to the same length."""
    max_len = min(max(item["input_ids"].shape[0] for item in batch), max_seq_len)

    input_ids    = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels       = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attn_mask    = torch.zeros(len(batch), max_len, dtype=torch.long)
    safety_mask  = torch.zeros(len(batch), max_len, dtype=torch.float)

    for i, item in enumerate(batch):
        L = min(item["input_ids"].shape[0], max_len)
        input_ids[i, :L]   = item["input_ids"][:L]
        labels[i, :L]      = item["labels"][:L]
        attn_mask[i, :L]   = 1
        safety_mask[i, :L] = item["safety_mask"][:L].float()

    return {
        "input_ids":      input_ids,
        "labels":         labels,
        "attention_mask": attn_mask,
        "safety_mask":    safety_mask,
    }


# ── LR Scheduler ─────────────────────────────────────────────────────────────

class CosineWarmupScheduler:
    """
    Cosine annealing with linear warmup.
    lr = peak_lr × cosine_decay, after warmup_steps of linear ramp.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ):
        self.optimizer    = optimizer
        self.peak_lr      = peak_lr
        self.min_lr       = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self._step        = 0

    def get_lr(self) -> float:
        step = self._step
        if step < self.warmup_steps:
            return self.peak_lr * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine_decay * (self.peak_lr - self.min_lr)

    def step(self):
        self._step += 1
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @property
    def current_lr(self) -> float:
        return self.get_lr()


# ── Trainer ───────────────────────────────────────────────────────────────────

class MindBridgeTrainer:
    """
    Orchestrates pre-training with FSDP, mixed precision, and clinical safety.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.model_cfg  = model_config
        self.train_cfg  = training_config
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main    = local_rank == 0
        self.device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self._setup_logging()
        self.model     = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler    = GradScaler(enabled=training_config.fp16)

        # State
        self.global_step    = 0
        self.tokens_consumed = 0

    # ── Setup ──────────────────────────────────────────────────────────────

    def _setup_logging(self):
        if self.is_main:
            logging.basicConfig(
                format="%(asctime)s | %(levelname)s | %(message)s",
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(
                        Path(self.train_cfg.logging_dir) / "pretrain.log"
                    ),
                ],
            )
            if WANDB_AVAILABLE:
                wandb.init(
                    project="mindbridge",
                    name=self.train_cfg.run_name,
                    config={**asdict(self.model_cfg), **asdict(self.train_cfg)},
                )

    def _build_model(self) -> torch.nn.Module:
        if self.is_main:
            logger.info(f"Building {self.model_cfg.model_name} "
                        f"({self.model_cfg.num_parameters()/1e9:.2f}B params)")

        model = MindBridgeLLM(self.model_cfg)

        # Gradient checkpointing — ~30% memory saving for ~20% speed cost
        if self.train_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable() if hasattr(model, "gradient_checkpointing_enable") \
                else self._apply_gradient_checkpointing(model)

        # FSDP wrap
        if self.train_cfg.fsdp and FSDP_AVAILABLE and self.world_size > 1:
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16 if self.train_cfg.bf16 else torch.float16,
                reduce_dtype=torch.bfloat16 if self.train_cfg.bf16 else torch.float16,
                buffer_dtype=torch.bfloat16 if self.train_cfg.bf16 else torch.float16,
            )
            auto_wrap = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={TransformerBlock},
            )
            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                auto_wrap_policy=auto_wrap,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                cpu_offload=CPUOffload(offload_params=self.train_cfg.fsdp_cpu_offload),
                device_id=self.local_rank,
            )
            if self.is_main:
                logger.info(f"FSDP enabled — sharding across {self.world_size} GPUs")
        else:
            model = model.to(self.device)
            if self.train_cfg.bf16 and torch.cuda.is_available():
                model = model.to(torch.bfloat16)

        return model

    def _apply_gradient_checkpointing(self, model: MindBridgeLLM):
        """Manual gradient checkpointing for TransformerBlocks."""
        from torch.utils.checkpoint import checkpoint

        def make_ckpt_forward(block):
            original_forward = block.forward

            def ckpt_forward(*args, **kwargs):
                return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

            block.forward = ckpt_forward

        for block in model.layers:
            make_ckpt_forward(block)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        AdamW with separate weight decay groups.
        Embeddings and norms have no weight decay (standard practice).
        """
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "embed" in name or param.ndim < 2:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = [
            {"params": decay_params,    "weight_decay": self.train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            groups,
            lr=self.train_cfg.learning_rate,
            betas=(self.train_cfg.adam_beta1, self.train_cfg.adam_beta2),
            eps=self.train_cfg.adam_epsilon,
            fused=torch.cuda.is_available(),  # fused kernel if on GPU
        )
        if self.is_main:
            n_decay = sum(p.numel() for p in decay_params)
            n_nodecay = sum(p.numel() for p in no_decay_params)
            logger.info(f"Optimizer: decay params={n_decay:,}  no-decay={n_nodecay:,}")
        return optimizer

    def _build_scheduler(self) -> CosineWarmupScheduler:
        return CosineWarmupScheduler(
            self.optimizer,
            peak_lr=self.train_cfg.learning_rate,
            min_lr=self.train_cfg.min_learning_rate,
            warmup_steps=self.train_cfg.warmup_steps,
            max_steps=self.train_cfg.max_steps,
        )

    # ── Data ──────────────────────────────────────────────────────────────────

    def _build_dataloaders(self):
        cfg = self.train_cfg

        train_ds = TokenizedDataset(cfg.train_data_dir, cfg.max_seq_len, split="train")
        eval_ds  = TokenizedDataset(cfg.eval_data_dir,  cfg.max_seq_len, split="eval")

        _collate = lambda b: collate_fn(b, self.model_cfg.pad_token_id, cfg.max_seq_len)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.per_device_train_batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=_collate,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=cfg.per_device_eval_batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=_collate,
        )
        return train_loader, eval_loader

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self):
        cfg = self.train_cfg
        train_loader, eval_loader = self._build_dataloaders()

        if self.is_main:
            logger.info(f"Starting pre-training: {cfg.run_name}")
            logger.info(f"  Max steps         : {cfg.max_steps:,}")
            logger.info(f"  Effective batch   : {cfg.effective_batch_size} seqs")
            logger.info(f"  Tokens/step       : {cfg.tokens_per_step:,}")

        self.model.train()
        accum_loss   = 0.0
        accum_steps  = 0
        t0           = time.time()

        self.optimizer.zero_grad()

        for batch in train_loader:
            if self.global_step >= cfg.max_steps:
                break

            # ── Forward ────────────────────────────────────────────────────
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                _, loss = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    safety_mask=batch["safety_mask"],
                )

            loss = loss / cfg.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            accum_loss  += loss.item()
            accum_steps += 1

            # ── Gradient accumulation step ──────────────────────────────────
            if accum_steps % cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step    += 1
                self.tokens_consumed += cfg.tokens_per_step
                step_loss = accum_loss * cfg.gradient_accumulation_steps
                accum_loss = 0.0

                # ── Logging ──────────────────────────────────────────────────
                if self.global_step % cfg.logging_steps == 0 and self.is_main:
                    elapsed = time.time() - t0
                    tokens_per_sec = cfg.tokens_per_step * cfg.logging_steps / elapsed
                    log_dict = {
                        "train/loss":       step_loss,
                        "train/perplexity": math.exp(min(step_loss, 20)),
                        "train/lr":         self.scheduler.current_lr,
                        "train/tokens":     self.tokens_consumed,
                        "perf/tokens_sec":  tokens_per_sec,
                        "perf/step":        self.global_step,
                    }
                    logger.info(
                        f"Step {self.global_step:6d} | "
                        f"loss {step_loss:.4f} | "
                        f"ppl {math.exp(min(step_loss,20)):.2f} | "
                        f"lr {self.scheduler.current_lr:.2e} | "
                        f"{tokens_per_sec/1e6:.1f}M tok/s"
                    )
                    if WANDB_AVAILABLE:
                        wandb.log(log_dict, step=self.global_step)
                    t0 = time.time()

                # ── Evaluation ────────────────────────────────────────────────
                if self.global_step % cfg.eval_steps == 0:
                    eval_loss = self._evaluate(eval_loader)
                    if self.is_main:
                        logger.info(f"  Eval loss: {eval_loss:.4f} | ppl: {math.exp(min(eval_loss,20)):.2f}")
                        if WANDB_AVAILABLE:
                            wandb.log({"eval/loss": eval_loss}, step=self.global_step)

                # ── Checkpoint ───────────────────────────────────────────────
                if self.global_step % cfg.save_steps == 0:
                    self._save_checkpoint()

        if self.is_main:
            logger.info(f"Training complete at step {self.global_step}")
            logger.info(f"Total tokens consumed: {self.tokens_consumed:,}")
        self._save_checkpoint(final=True)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, max_batches: int = 50) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.train_cfg.bf16):
                _, loss = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    safety_mask=batch["safety_mask"],
                )
            total_loss += loss.item()
            n += 1
        self.model.train()
        return total_loss / max(n, 1)

    def _save_checkpoint(self, final: bool = False):
        if not self.is_main:
            return
        tag = "final" if final else f"step_{self.global_step:06d}"
        ckpt_dir = Path(self.train_cfg.output_dir) / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # FSDP: gather shards before saving
        if isinstance(self.model, FSDP if FSDP_AVAILABLE else type(None)):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                state = self.model.state_dict()
        else:
            state = self.model.state_dict()

        torch.save({
            "model_state_dict":  state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step":       self.global_step,
            "tokens_consumed":   self.tokens_consumed,
            "model_config":      asdict(self.model_cfg),
            "training_config":   asdict(self.train_cfg),
        }, ckpt_dir / "checkpoint.pt")

        logger.info(f"Checkpoint saved → {ckpt_dir}")

        # Prune old checkpoints
        self._prune_checkpoints()

    def _prune_checkpoints(self):
        output_dir = Path(self.train_cfg.output_dir)
        ckpts = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        limit = self.train_cfg.save_total_limit
        for old in ckpts[:-limit]:
            import shutil
            shutil.rmtree(old)
            logger.info(f"Pruned checkpoint: {old}")
