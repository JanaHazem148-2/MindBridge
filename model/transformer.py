"""
model/transformer.py
─────────────────────
MindBridge GPT-style causal decoder transformer.

Architecture decisions:
  - RoPE positional embeddings (no learned positions, better length generalisation)
  - RMSNorm instead of LayerNorm (faster, no bias)
  - SwiGLU activation in FFN (matches LLaMA/Mistral family)
  - Grouped-Query Attention (GQA) optional — set num_key_value_heads < num_attention_heads
  - Pre-norm (norm before attention/FFN, not after) — more stable training
  - No bias in linear projections (matches modern practice)

Dependencies:
    torch >= 2.0
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# local
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from configs.model_config import ModelConfig


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square normalisation.
    Simpler than LayerNorm (no mean subtraction, no bias) and marginally faster.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


# ── Rotary Positional Embeddings (RoPE) ──────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    RoPE: apply position-dependent rotation to Q and K vectors.
    Position information is baked into dot-products rather than added to tokens.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10_000.0):
        super().__init__()
        # inv_freq: [dim/2] — precompute the frequency denominators
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # [seq_len, dim/2]
        emb   = torch.cat([freqs, freqs], dim=-1)       # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension by 90°."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


# ── Grouped-Query Attention ───────────────────────────────────────────────────

class ClinicalAttention(nn.Module):
    """
    Multi-head or Grouped-Query causal self-attention.

    When config.num_key_value_heads < config.num_attention_heads,
    KV heads are repeated (GQA) — reduces KV cache memory.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size     = config.hidden_size
        self.num_heads       = config.num_attention_heads
        self.num_kv_heads    = config.num_key_value_heads
        self.head_dim        = config.hidden_size // config.num_attention_heads
        self.num_kv_groups   = self.num_heads // self.num_kv_heads

        # Projections — no bias (LLaMA-style)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size,    bias=False)

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.attn_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,                  # [B, S, H]
        attention_mask: Optional[torch.Tensor] = None, # [B, 1, S, S]
        past_key_value: Optional[Tuple] = None,        # KV cache for inference
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:

        B, S, _ = hidden_states.shape

        # Project
        q = self.q_proj(hidden_states).view(B, S, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Append to KV cache if provided (incremental decoding)
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        seq_len = k.shape[2]  # after cache concat
        cos, sin = self.rotary_emb(q, seq_len=seq_len)

        # Apply RoPE only to the last S positions
        cos_q = cos[:, :, -S:, :]
        sin_q = sin[:, :, -S:, :]
        q, k = apply_rotary_emb(q, k, cos_q, sin_q)

        # GQA: repeat KV heads to match Q head count
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention (uses Flash Attention when available)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=(attention_mask is None),  # use causal mask if no explicit mask
            scale=scale,
        )

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out), present


# ── SwiGLU Feed-Forward Network ───────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN: two parallel linear projections, gated by SiLU.
    Slightly better quality than ReLU/GELU at the same parameter count.
    FFN hidden dim is typically 8/3 × hidden_size, rounded to nearest 256.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn    = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) × up, then project down
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single pre-norm transformer layer:
      x → RMSNorm → Attention → residual
        → RMSNorm → FFN       → residual
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm       = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn             = ClinicalAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp                   = SwiGLUFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:

        # Self-attention with residual
        residual = hidden_states
        hidden_states, present = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, present


# ── Full Model ────────────────────────────────────────────────────────────────

class MindBridgeLLM(nn.Module):
    """
    MindBridge GPT-style decoder LLM.

    forward() returns (logits, loss) where loss is only computed when
    labels are provided. For pre-training, labels = input_ids shifted by 1.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                          padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm   = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying is explicitly off (config.tie_word_embeddings=False)
        # because our lm_head benefits from a separate projection

        self._init_weights()

    def _init_weights(self):
        """
        Standard GPT-style init:
        - Linear and Embedding: normal with std = initializer_range
        - Residual projections scaled by 1/sqrt(2 * num_layers)
        """
        std = self.config.initializer_range
        residual_scale = 1.0 / math.sqrt(2.0 * self.config.num_hidden_layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Scale residual projections (o_proj and down_proj)
        for block in self.layers:
            nn.init.normal_(block.self_attn.o_proj.weight, mean=0.0, std=std * residual_scale)
            nn.init.normal_(block.mlp.down_proj.weight,    mean=0.0, std=std * residual_scale)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,                         # [B, S]
        attention_mask: Optional[torch.Tensor] = None,   # [B, S]
        labels: Optional[torch.Tensor] = None,           # [B, S] — same as input_ids for CLM
        past_key_values: Optional[list] = None,          # List of (k, v) per layer
        use_cache: bool = False,
        safety_mask: Optional[torch.Tensor] = None,      # [B, S] — 1 = exclude from loss
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, S = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)     # [B, S, H]

        # Build causal 4D mask from 2D padding mask (if provided)
        causal_mask = None
        if attention_mask is not None:
            # Expand [B, S] → [B, 1, S, S] for SDPA
            # (SDPA with is_causal=False needs explicit causal mask)
            causal_mask = _make_causal_mask(attention_mask, hidden_states.dtype)

        # Run through transformer layers
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                present_key_values.append(present)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()     # [B, S, vocab_size]

        loss = None
        if labels is not None:
            # Causal LM loss: predict token[i+1] from token[i]
            shift_logits = logits[:, :-1, :].contiguous()   # [B, S-1, V]
            shift_labels = labels[:, 1:].contiguous()        # [B, S-1]

            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.config.pad_token_id,
                reduction="none",
            )
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Apply safety mask: zero out loss on safety-flagged tokens
            # Safety tokens contribute to context but not to the gradient target
            if safety_mask is not None:
                shift_safety = safety_mask[:, 1:].contiguous().view(-1).float()
                per_token_loss = per_token_loss * (1.0 - shift_safety)

            # Average over non-padded, non-masked tokens
            valid_tokens = (shift_labels.view(-1) != self.config.pad_token_id).float()
            if safety_mask is not None:
                valid_tokens = valid_tokens * (1.0 - shift_safety)

            loss = per_token_loss.sum() / valid_tokens.sum().clamp(min=1.0)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Simple greedy / nucleus sampling generation.
        Production inference should use vLLM or TGI instead.
        """
        device = input_ids.device
        past_key_values = None

        for _ in range(max_new_tokens):
            logits, _ = self.forward(
                input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            # Temperature scaling
            next_logits = logits[:, -1, :] / max(temperature, 1e-9)

            # Top-p (nucleus) sampling
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumprob = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprob - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            probs = F.softmax(sorted_logits.scatter(1, sorted_idx, sorted_logits), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break

        return input_ids

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)


# ── Utility: causal mask builder ─────────────────────────────────────────────

def _make_causal_mask(
    attention_mask: torch.Tensor,  # [B, S]
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert [B, S] padding mask to [B, 1, S, S] causal + padding mask
    suitable for scaled_dot_product_attention (additive mask, -inf for masked).
    """
    B, S = attention_mask.shape
    device = attention_mask.device

    # Lower-triangular causal mask [1, 1, S, S]
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)

    # Padding mask [B, 1, 1, S]
    pad = attention_mask[:, None, None, :].bool()

    # Combined: causal AND not-padded
    combined = causal & pad

    # Convert to additive mask
    mask = torch.zeros(B, 1, S, S, dtype=dtype, device=device)
    mask.masked_fill_(~combined, float("-inf"))
    return mask


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch = __import__("torch")

    cfg = ModelConfig.tiny()
    print(f"Building {cfg.model_name}...")
    model = MindBridgeLLM(cfg)
    n = model.num_parameters()
    print(f"Parameters: {n:,}  ({n/1e6:.1f}M)")

    # Forward pass
    B, S = 2, 64
    ids    = torch.randint(0, cfg.vocab_size, (B, S))
    mask   = torch.ones(B, S, dtype=torch.long)
    mask[0, 50:] = 0  # simulate padding

    logits, loss = model(ids, attention_mask=mask, labels=ids)
    print(f"Logits shape : {logits.shape}")
    print(f"Loss         : {loss.item():.4f}")
    print(f"\n✓ Forward pass OK")

    # Test safety mask
    safety = torch.zeros(B, S)
    safety[1, 30:40] = 1.0   # flag 10 tokens as safety-restricted
    _, loss_safe = model(ids, attention_mask=mask, labels=ids, safety_mask=safety)
    print(f"Loss w/ safety mask: {loss_safe.item():.4f}")
    print(f"✓ Safety mask OK")
