"""
scripts/pretrain.py
───────────────────
Entry point for pre-training MindBridgeLLM.

Single GPU (dev):
    python scripts/pretrain.py --model tiny --no_fsdp

Multi-GPU (8× A100):
    torchrun --nproc_per_node=8 --master_port=29500 scripts/pretrain.py --model 1b

Resume from checkpoint:
    torchrun --nproc_per_node=8 scripts/pretrain.py --model 1b --resume checkpoints/step_010000
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from configs.model_config import ModelConfig
from configs.training_config import TrainingConfig
from training.trainer import MindBridgeTrainer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="MindBridge LLM Pre-training")

    # Model size
    parser.add_argument(
        "--model", type=str, default="tiny",
        choices=["tiny", "1b", "7b"],
        help="Model size preset (tiny=125M, 1b=1.3B, 7b=7B)",
    )

    # Training overrides
    parser.add_argument("--max_steps",       type=int,   default=None)
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--output_dir",      type=str,   default="./checkpoints")
    parser.add_argument("--train_data_dir",  type=str,   default="./data/tokenized/train")
    parser.add_argument("--eval_data_dir",   type=str,   default="./data/tokenized/eval")
    parser.add_argument("--run_name",        type=str,   default=None)

    # Flags
    parser.add_argument("--no_fsdp",  action="store_true", help="Disable FSDP (single GPU)")
    parser.add_argument("--no_bf16",  action="store_true", help="Disable bfloat16")
    parser.add_argument("--resume",   type=str, default=None, help="Path to checkpoint dir")

    return parser.parse_args()


def setup_distributed():
    """Initialise the process group for distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    else:
        return 0, 1


def load_checkpoint_configs(checkpoint_path: str):
    """Load model + training config from a saved checkpoint."""
    ckpt = torch.load(os.path.join(checkpoint_path, "checkpoint.pt"), map_location="cpu")
    model_cfg    = ModelConfig(**ckpt["model_config"])
    training_cfg = TrainingConfig(**ckpt["training_config"])
    return model_cfg, training_cfg, ckpt["global_step"]


def main():
    args = parse_args()

    # ── Distributed setup ────────────────────────────────────────────────────
    local_rank, world_size = setup_distributed()
    is_main = local_rank == 0

    # ── Build configs ────────────────────────────────────────────────────────
    if args.resume:
        if is_main:
            print(f"Resuming from checkpoint: {args.resume}")
        model_cfg, train_cfg, resume_step = load_checkpoint_configs(args.resume)
    else:
        # Fresh run
        model_cfg_map = {"tiny": ModelConfig.tiny, "1b": ModelConfig.base_1b, "7b": ModelConfig.large_7b}
        model_cfg = model_cfg_map[args.model]()
        train_cfg = TrainingConfig()
        resume_step = 0

    # Apply CLI overrides
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        train_cfg.per_device_train_batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
    if args.output_dir:
        train_cfg.output_dir = args.output_dir
    if args.train_data_dir:
        train_cfg.train_data_dir = args.train_data_dir
    if args.eval_data_dir:
        train_cfg.eval_data_dir = args.eval_data_dir
    if args.no_fsdp:
        train_cfg.fsdp = False
    if args.no_bf16:
        train_cfg.bf16 = False
    if args.run_name:
        train_cfg.run_name = args.run_name
    else:
        train_cfg.run_name = f"mindbridge-{args.model}-pretrain"

    # Create output dirs
    if is_main:
        os.makedirs(train_cfg.output_dir, exist_ok=True)
        os.makedirs(train_cfg.logging_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  MindBridge LLM Pre-training")
        print(f"{'='*60}")
        print(f"  Model    : {model_cfg.model_name}")
        print(f"  Params   : {model_cfg.num_parameters()/1e9:.2f}B")
        print(f"  World    : {world_size} GPU(s)")
        print(f"  Max steps: {train_cfg.max_steps:,}")
        print(f"  Batch    : {train_cfg.per_device_train_batch_size} × {world_size} GPUs "
              f"× {train_cfg.gradient_accumulation_steps} accum = "
              f"{train_cfg.effective_batch_size} seqs / step")
        print(f"  Tokens/step: {train_cfg.tokens_per_step:,}")
        print(f"  Data     : {train_cfg.train_data_dir}")
        print(f"  Output   : {train_cfg.output_dir}")
        print(f"{'='*60}\n")

    # ── Build trainer and run ────────────────────────────────────────────────
    trainer = MindBridgeTrainer(
        model_config=model_cfg,
        training_config=train_cfg,
        local_rank=local_rank,
        world_size=world_size,
    )

    # Resume: fast-forward scheduler to resume_step
    if resume_step > 0 and args.resume:
        trainer.global_step = resume_step
        for _ in range(resume_step):
            trainer.scheduler.step()
        ckpt = torch.load(
            os.path.join(args.resume, "checkpoint.pt"),
            map_location=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.tokens_consumed = ckpt["tokens_consumed"]
        if is_main:
            print(f"Resumed from step {resume_step}, tokens={trainer.tokens_consumed:,}")

    trainer.train()

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
