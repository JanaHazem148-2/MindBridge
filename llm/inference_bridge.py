"""
llm/inference_bridge.py
────────────────────────
Drop-in replacement for MindBridgeLLMInference in agents/agent_orchestrator.py.

Routes to Groq's Chat Completions API (OpenAI-compatible endpoint).

Priority chain:
  1. Groq API  (primary — reads GROQ_API_KEY)
  2. Safe stub  (if key not configured)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


class InferenceBridge:
    """
    Unified inference interface — same public API as MindBridgeLLMInference.

    Parameters
    ----------
    use_openai      : Kept for backward-compat (always True, routes to Groq)
    model_path      : Ignored (Groq is cloud-only)
    api_key         : Groq API key (falls back to GROQ_API_KEY env var)
    model           : Model string  (falls back to config.GROQ_MODEL)
    api_base        : API base URL  (falls back to config.GROQ_API_BASE)
    max_new_tokens, temperature, top_p : Generation hyperparameters
    fallback_to_stub : Return stub on failure instead of raising
    """

    def __init__(
        self,
        use_openai:       bool  = True,
        model_path:       Optional[str] = None,
        api_key:          Optional[str] = None,
        api_base:         Optional[str] = None,
        model:            Optional[str] = None,
        max_new_tokens:   int   = 1024,
        temperature:      float = 0.7,
        top_p:            float = 0.9,
        fallback_to_stub: bool  = True,
    ):
        self.use_openai     = use_openai
        self.model_path     = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p
        self._loaded        = False
        self.backend        = "unset"

        self._groq_llm = None

        self._init(api_key, api_base, model, fallback_to_stub)

    def _init(self, api_key, api_base, model, fallback_to_stub):
        import config as cfg

        resolved_key   = api_key  or os.environ.get("GROQ_API_KEY", "") or cfg.GROQ_API_KEY
        resolved_base  = api_base or cfg.GROQ_API_BASE
        resolved_model = model    or cfg.GROQ_MODEL

        if resolved_key:
            try:
                from llm.orchestrator import LLMOrchestrator
                self._groq_llm = LLMOrchestrator(
                    api_key          = resolved_key,
                    api_base         = resolved_base,
                    model            = resolved_model,
                    max_tokens       = self.max_new_tokens,
                    temperature      = self.temperature,
                    top_p            = self.top_p,
                    fallback_to_stub = fallback_to_stub,
                )
                self.backend = f"groq:{resolved_model}"
                self.model   = resolved_model
                self._loaded = True
                logger.info(f"InferenceBridge → Groq | model={resolved_model}")
            except Exception as e:
                logger.error(f"Failed to init Groq LLM: {e}")
                self.backend = "stub"
                self.model   = "stub"
        else:
            logger.warning(
                "GROQ_API_KEY not set — InferenceBridge using stub.\n"
                "Add GROQ_API_KEY=gsk_... to your .env file."
            )
            self.backend = "stub"
            self.model   = "stub"

    def generate(
        self,
        system_prompt: str,
        conversation:  list[dict],
        context_block: str = "",
        extra_system:  str = "",
    ) -> str:
        if self._groq_llm:
            return self._groq_llm.generate(
                system_prompt = system_prompt,
                conversation  = conversation,
                context_block = context_block,
                extra_system  = extra_system,
            )
        return self._stub(conversation)

    def _stub(self, conversation: list[dict]) -> str:
        last = next(
            (t["content"] for t in reversed(conversation) if t["role"] == "user"),
            "(no message)"
        )
        return (
            "[MindBridge — GROQ_API_KEY not configured]\n\n"
            f"I received: \"{last[:80]}\"\n\n"
            "Add your key to .env:  GROQ_API_KEY=gsk_...\n"
            "Get a free key at: https://console.groq.com/keys\n\n"
            "If you are in crisis right now, call 08008880700 (Egypt, free, 24/7)."
        )
