"""
configs/model_config.py
───────────────────────
MindBridge GPT-style decoder config.
Three size presets: tiny (dev), 1.3B (production), 7B (research).
All sizes share the same architecture code — only numbers change.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # ── Vocabulary ──────────────────────────────────────────────────────────
    vocab_size: int = 32_000          # BPE tokenizer trained on clinical corpus
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Sequence length ──────────────────────────────────────────────────────
    max_seq_len: int = 2048            # Clinical dialogues rarely exceed 2K tokens

    # ── Transformer dimensions ───────────────────────────────────────────────
    hidden_size: int = 2048
    intermediate_size: int = 8192      # FFN hidden dim = 4 × hidden_size
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16      # Set < num_attention_heads for GQA

    # ── Positional encoding ──────────────────────────────────────────────────
    rope_theta: float = 10_000.0       # RoPE base frequency
    rope_scaling: Optional[dict] = None  # e.g. {"type": "linear", "factor": 2.0}

    # ── Normalization ────────────────────────────────────────────────────────
    rms_norm_eps: float = 1e-5         # RMSNorm epsilon (LLaMA-style, no bias)

    # ── Dropout (0 during pre-training, light during SFT) ───────────────────
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # ── Activation ───────────────────────────────────────────────────────────
    hidden_act: str = "silu"           # SwiGLU gate activation

    # ── Initialisation ───────────────────────────────────────────────────────
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False   # Separate lm_head weights

    # ── Clinical domain metadata (stored in config, not used in forward pass) ─
    model_name: str = "mindbridge-1.3b"
    domain: str = "clinical-mental-health"
    safety_version: str = "v1.0"

    def __post_init__(self):
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads"
        )
        assert self.hidden_size % self.num_attention_heads == 0, (
            "hidden_size must be divisible by num_attention_heads"
        )

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """
        ~125M params — for fast dev/test iteration on a single GPU.
        Matches GPT-2 small in rough scale.
        """
        return cls(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
            max_seq_len=1024,
            model_name="mindbridge-tiny",
        )

    @classmethod
    def base_1b(cls) -> "ModelConfig":
        """
        ~1.3B params — primary production target.
        Requires ~24GB VRAM per GPU at bf16; fits comfortably on 8× A100 40GB.
        """
        return cls(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
            max_seq_len=2048,
            model_name="mindbridge-1.3b",
        )

    @classmethod
    def large_7b(cls) -> "ModelConfig":
        """
        ~7B params — research / highest quality.
        Requires 8× A100 80GB with FSDP ZeRO-3.
        """
        return cls(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,      # GQA: 4 queries per KV head
            max_seq_len=4096,
            model_name="mindbridge-7b",
        )

    def num_parameters(self) -> int:
        """Rough parameter count estimate."""
        emb = self.vocab_size * self.hidden_size
        per_layer = (
            # Attention: Q,K,V,O projections
            4 * self.hidden_size * self.hidden_size
            # FFN: gate, up, down (SwiGLU has 3 matrices)
            + 3 * self.hidden_size * self.intermediate_size
            # RMSNorm × 2 per layer
            + 2 * self.hidden_size
        )
        lm_head = self.vocab_size * self.hidden_size
        total = emb + per_layer * self.num_hidden_layers + lm_head
        return total
