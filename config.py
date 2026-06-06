"""
config.py
──────────
MindBridge configuration — Groq API key and model settings.

Groq runs open-source LLMs on custom LPU hardware, delivering
ultra-low-latency inference (typically < 300ms for a full reply).

Quick setup — create a .env file next to config.py:
    GROQ_API_KEY=gsk_...              ← your Groq key (console.groq.com/keys)
    GROQ_MODEL=llama-3.3-70b-versatile  ← or any model at console.groq.com/docs/models

Available Groq models (May 2026):
    llama-3.3-70b-versatile          ← best quality, recommended
    llama3-70b-8192                  ← good quality, 8k context
    llama3-8b-8192                   ← fastest, 8k context
    mixtral-8x7b-32768               ← long context (32k), fast
    gemma2-9b-it                     ← Google Gemma 2
"""

import os
from pathlib import Path

# ── Load .env if present ──────────────────────────────────────────────────────
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

# ── Groq API ──────────────────────────────────────────────────────────────────
# Your Groq key — get it from https://console.groq.com/keys
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

# Model slug — see https://console.groq.com/docs/models
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# Groq base URL — OpenAI-SDK compatible endpoint
GROQ_API_BASE: str = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")

# ── Backwards-compat aliases (used by legacy code paths) ──────────────────────
OPENAI_API_KEY: str = GROQ_API_KEY
OPENAI_MODEL:   str = GROQ_MODEL
OPENAI_API_BASE: str = GROQ_API_BASE

# ── Region & language ─────────────────────────────────────────────────────────
REGION:        str  = os.environ.get("MINDBRIDGE_REGION", "EG")
ENABLE_ARABIC: bool = os.environ.get("MINDBRIDGE_ARABIC", "1") == "1"

# ── Generation defaults ───────────────────────────────────────────────────────
MAX_TOKENS:  int   = int(os.environ.get("MINDBRIDGE_MAX_TOKENS", "1024"))
TEMPERATURE: float = float(os.environ.get("MINDBRIDGE_TEMPERATURE", "0.7"))
TOP_P:       float = float(os.environ.get("MINDBRIDGE_TOP_P", "0.9"))

# ── Paths ─────────────────────────────────────────────────────────────────────
RAG_DIR:           str = os.environ.get("MINDBRIDGE_RAG_DIR", "")
MEMORY_DIR:        str = os.environ.get("MINDBRIDGE_MEMORY_DIR", "/tmp/mindbridge_memory")
CLASSIFIER_PATH:   str = os.environ.get("MINDBRIDGE_CLASSIFIER_PATH", "")
CLINICIAN_WEBHOOK: str = os.environ.get("MINDBRIDGE_WEBHOOK", "")

# ── Validation ────────────────────────────────────────────────────────────────

def validate() -> None:
    """Raise if minimum required config is missing."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n\n"
            "Create a .env file next to config.py with:\n"
            "    GROQ_API_KEY=gsk_...                  <- your Groq key\n"
            "    GROQ_MODEL=llama-3.3-70b-versatile    <- optional, this is the default\n\n"
            "Get your free key at: https://console.groq.com/keys"
        )

def summary() -> dict:
    """Return a safe (no secrets) config summary for logging."""
    return {
        "model":          GROQ_MODEL,
        "api_base":       GROQ_API_BASE,
        "region":         REGION,
        "arabic_enabled": ENABLE_ARABIC,
        "max_tokens":     MAX_TOKENS,
        "temperature":    TEMPERATURE,
        "rag_dir":        RAG_DIR or "(not set)",
        "memory_dir":     MEMORY_DIR,
        "api_key_set":    bool(GROQ_API_KEY),
    }

# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    print("MindBridge Configuration (Groq)")
    print("=" * 40)
    print(json.dumps(summary(), indent=2))
    try:
        validate()
        print("\n✅  API key present — ready to run")
    except EnvironmentError as e:
        print(f"\n⚠️  {e}")
