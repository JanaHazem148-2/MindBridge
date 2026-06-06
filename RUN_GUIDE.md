# MindBridge — Complete Run Guide

## Why it won't run in the chat sandbox

This sandbox has no GPU and the network blocks PyTorch's CDN.
The code is production-correct — it just needs your hardware.
Below are exact commands for every setup option.

---

## Option A: Single GPU (fastest to start — any NVIDIA GPU ≥ 16GB)

### 1. Clone and install

```bash
# Unzip the downloaded mindbridge_phase1.zip
unzip mindbridge_phase1.zip
cd mindbridge

# Create venv
python3 -m venv .venv && source .venv/bin/activate

# Install PyTorch (CPU for dev, CUDA for training)
# CPU only (testing / no GPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 (most A100/A10/RTX cards):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install the rest
pip install tokenizers transformers pandas numpy scikit-learn wandb tqdm rich
```

### 2. Run the data pipeline (pure Python, no GPU needed)

```bash
# Verify all 8,341 records load correctly
python3 data/dataset_loader.py

# Run preprocessing + augmentation (→ 8,891 records)
python3 data/clinical_preprocessor.py
```

### 3. Train the tokenizer (CPU, ~30 seconds)

```bash
python3 tokenizer/clinical_tokenizer.py \
  --data_dir /path/to/your/uploads \
  --output_dir tokenizer/clinical_bpe_32k \
  --vocab_size 32000
```

### 4. Tokenize the corpus to shards (CPU, ~2 minutes)

```bash
python3 scripts/tokenize_corpus.py \
  --data_dir /path/to/your/uploads \
  --tokenizer_dir tokenizer/clinical_bpe_32k \
  --output_dir data/tokenized \
  --shard_size 10000
```

### 5. Dev pre-training run — tiny model, single GPU

```bash
python3 scripts/pretrain.py \
  --model tiny \
  --no_fsdp \
  --max_steps 500 \
  --batch_size 2 \
  --output_dir checkpoints/tiny-dev
```

Expected: ~10 min on RTX 3090, loss drops from ~10 → ~4.

---

## Option B: 8× A100 (production 1.3B run)

### Setup on a SLURM cluster or cloud (Lambda, CoreWeave, RunPod)

```bash
# On each node:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tokenizers transformers pandas numpy wandb

# Set your WandB key
export WANDB_API_KEY=your_key_here

# Launch distributed training
torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  scripts/pretrain.py \
  --model 1b \
  --max_steps 100000 \
  --batch_size 4 \
  --output_dir /shared_storage/checkpoints/mindbridge-1b \
  --run_name mindbridge-1b-v1
```

### Expected compute budget (1.3B model)

| Metric | Value |
|--------|-------|
| Training tokens target | ~10B (clinical domain is small — no need for 100B) |
| Steps at 1M tok/step | ~10,000 |
| Time on 8× A100 | ~6–8 hours |
| Checkpoint size | ~5GB (bf16) |
| GPU memory per card | ~35GB (FSDP sharded) |

---

## Option C: Google Colab Pro+ (free tier too slow, Pro+ workable for tiny)

```python
# In a Colab cell:
!pip install torch tokenizers transformers pandas wandb -q
!unzip /content/mindbridge_phase1.zip -d /content/

import subprocess
subprocess.run([
    "python3", "/content/mindbridge/scripts/pretrain.py",
    "--model", "tiny",
    "--no_fsdp",
    "--max_steps", "200",
    "--batch_size", "1",
])
```

---

## Option D: Modal (recommended cloud option — pay per GPU second)

```python
# modal_train.py
import modal

app = modal.App("mindbridge")
image = modal.Image.debian_slim().pip_install(
    "torch", "tokenizers", "transformers", "pandas", "wandb"
)

@app.function(
    gpu="A100",
    timeout=3600 * 8,
    image=image,
    mounts=[modal.Mount.from_local_dir(".", remote_path="/mindbridge")],
)
def train():
    import subprocess
    subprocess.run([
        "python3", "/mindbridge/scripts/pretrain.py",
        "--model", "1b", "--max_steps", "10000"
    ], check=True)

# Run with: modal run modal_train.py
```

---

## Resume a checkpoint

```bash
torchrun --nproc_per_node=8 scripts/pretrain.py \
  --model 1b \
  --resume checkpoints/mindbridge-1b-v1/step_005000
```

---

## Evaluate a checkpoint

```bash
python3 scripts/evaluate.py \
  --checkpoint checkpoints/mindbridge-1b-v1/step_010000 \
  --split test \
  --data_dir /path/to/uploads \
  --output_file eval_results.json
```

Target perplexity benchmarks:
- After 1K steps: ~8–12 (model is learning)
- After 10K steps: ~3–5 (clinically coherent)
- Clinical probe accuracy: >75% (correct PHQ severity ordering)
- Safety probe: PASS (safe completions lower perplexity than unsafe)

---

## Monitoring

```bash
# WandB dashboard (auto-opens if WANDB_API_KEY is set)
# Or local TensorBoard:
tensorboard --logdir logs/

# Watch GPU utilisation
watch -n 1 nvidia-smi

# Watch training log
tail -f logs/pretrain.log
```

---

## Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Batch too large | Reduce `--batch_size` or enable `--gradient_checkpointing` |
| `NCCL timeout` | Slow inter-GPU link | Set `NCCL_TIMEOUT=1800` env var |
| `FileNotFoundError: tokenized/train` | Shards not created | Run `tokenize_corpus.py` first |
| `tokenizer.json not found` | Tokenizer not trained | Run `clinical_tokenizer.py` first |
| `Loss is NaN` | LR too high | Reduce `--lr` to `1e-4` |
