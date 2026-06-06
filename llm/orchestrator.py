"""
llm/orchestrator.py
────────────────────
MindBridge LLM Orchestrator — Groq SDK backend.

Uses the official groq Python SDK which handles auth headers correctly
and avoids Cloudflare bot-detection that blocks raw urllib requests.

Install: pip install groq
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

logger = logging.getLogger(__name__)


def _call_groq(messages, model, max_tokens, temperature, top_p, api_key):
    """Call Groq — tries SDK first, falls back to requests."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return completion.choices[0].message.content
    except ImportError:
        pass  # SDK not installed, fall through to requests

    # Fallback: plain requests (OpenAI-compatible endpoint)
    import requests as _req
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    resp = _req.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


class LLMOrchestrator:
    def __init__(
        self,
        api_key:          Optional[str] = None,
        model:            Optional[str] = None,
        api_base:         Optional[str] = None,   # kept for compat, unused
        max_tokens:       Optional[int]   = None,
        temperature:      Optional[float] = None,
        top_p:            Optional[float] = None,
        fallback_to_stub: bool = False,
    ):
        self.api_key          = api_key     or config.GROQ_API_KEY
        self.model            = model       or config.GROQ_MODEL
        self.max_tokens       = max_tokens  or config.MAX_TOKENS
        self.temperature      = temperature if temperature is not None else config.TEMPERATURE
        self.top_p            = top_p       if top_p       is not None else config.TOP_P
        self.fallback_to_stub = fallback_to_stub

        if not self.api_key:
            if fallback_to_stub:
                logger.warning("No GROQ_API_KEY — will use stub responses")
            else:
                raise EnvironmentError("GROQ_API_KEY not set.")

        logger.info(
            f"LLMOrchestrator ready | model={self.model} | "
            f"api_base=https://api.groq.com/openai/v1 | max_tokens={self.max_tokens}"
        )

    def generate(self, system_prompt, conversation, context_block="", extra_system=""):
        t_start = time.time()

        parts = [system_prompt]
        if context_block:
            parts.append(context_block)
        if extra_system:
            parts.append(extra_system)
        system_content = "\n\n".join(parts)

        messages = [{"role": "system", "content": system_content}]
        messages += conversation[-12:]

        if not self.api_key:
            return self._stub(conversation)

        try:
            text = _call_groq(
                messages    = messages,
                model       = self.model,
                max_tokens  = self.max_tokens,
                temperature = self.temperature,
                top_p       = self.top_p,
                api_key     = self.api_key,
            )
            elapsed = round((time.time() - t_start) * 1000, 1)
            logger.debug(f"Groq call OK in {elapsed}ms | model={self.model}")
            return text
        except RuntimeError as e:
            logger.error(f"Groq error: {e}")
            if self.fallback_to_stub:
                return self._stub(conversation)
            raise

    def summarise(self, session_turns):
        history = "\n".join(
            f"{'User' if t['role'] == 'user' else 'MindBridge'}: {t['content']}"
            for t in session_turns[-20:]
        )
        system = (
            "You are a clinical documentation assistant for MindBridge.\n"
            "Summarise the therapy session in 3-5 sentences.\n"
            "Include: main themes, emotional tone, PHQ/mood scores, goals set, "
            "coping strategies discussed, and any safety flags.\n"
            "Be concise, clinical, third-person. No PII."
        )
        return self.generate(system_prompt=system, conversation=[{"role": "user", "content": history}])

    def score_safety(self, text, context=""):
        system = (
            "You are a clinical safety monitor for a mental health AI.\n"
            "Assess the user message for crisis indicators.\n\n"
            "Reply ONLY with valid JSON — no markdown fences:\n"
            '{"level":"safe"|"monitor"|"soft_intervene"|"hard_escalate",'
            '"confidence":0.0-1.0,"reason":"<one sentence>"}\n\n'
            "Levels:\n"
            "  safe            — no distress signals\n"
            "  monitor         — mild distress, flag for review\n"
            "  soft_intervene  — moderate distress, redirect gently\n"
            "  hard_escalate   — suicidal ideation / self-harm / immediate danger"
        )
        note = f"\nContext: {context}" if context else ""
        raw = self.generate(system_prompt=system, conversation=[{"role": "user", "content": text + note}])
        try:
            clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.warning(f"Safety score parse error: {raw[:120]}")
            return {"level": "monitor", "confidence": 0.5, "reason": "parse error — defaulting to monitor"}

    def generate_ritual(self, user_context, mood_score):
        system = (
            "You are the MindBridge Ritual Planner.\n"
            "Propose ONE personalised therapy ritual based on the user's current state.\n\n"
            "Ritual types: guided breathing, body scan, journaling prompt, "
            "gratitude practice, 5-4-3-2-1 grounding, progressive muscle relaxation.\n\n"
            "Output format (markdown):\n"
            "**Ritual**: <name>\n"
            "**Why**: <one sentence linking to their state>\n"
            "**Steps**: (3-5 warm, accessible steps)\n"
            "**Duration**: <X minutes>\n"
            "**After**: <prompt to log mood>"
        )
        return self.generate(
            system_prompt=system,
            conversation=[{"role": "user", "content": f"Mood: {mood_score}/10\nContext: {user_context}"}],
        )

    def ifs_moderate(self, parts_context, user_message):
        system = (
            "You are the MindBridge IFS Facilitator moderating a dialogue "
            "between the user's internal parts.\n\n"
            f"Parts active in this session:\n{parts_context}\n\n"
            "Your role:\n"
            "- Name which part seems to be speaking\n"
            "- Validate each part's positive intention\n"
            "- Help the user's calm Self witness without merging\n"
            "- Guide towards integration of conflicting voices\n\n"
            "Tone: curious, warm, never pathologising."
        )
        return self.generate(system_prompt=system, conversation=[{"role": "user", "content": user_message}])

    def health_check(self):
        if not self.api_key:
            return {"status": "no_api_key", "model": self.model}
        try:
            resp = self.generate(
                system_prompt="Reply with exactly the word: OK",
                conversation=[{"role": "user", "content": "health check"}],
            )
            ok = "ok" in resp.strip().lower()
            return {"status": "ok" if ok else "unexpected_response", "response": resp[:60], "model": self.model}
        except Exception as e:
            return {"status": "error", "error": str(e), "model": self.model}

    def _stub(self, conversation):
        last = next((t["content"] for t in reversed(conversation) if t["role"] == "user"), "(no message)")
        return (
            "[MindBridge — GROQ_API_KEY not configured]\n\n"
            f"I received: \"{last[:80]}\"\n\n"
            "Add your key to .env:  GROQ_API_KEY=gsk_...\n"
            "Get a free key at: https://console.groq.com/keys\n\n"
            "If you are in crisis right now, call 08008880700 (Egypt, free, 24/7)."
        )


if __name__ == "__main__":
    import json as _json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print("=" * 60)
    print("  LLMOrchestrator — Groq Connection Test")
    print("=" * 60)
    try:
        config.validate()
    except EnvironmentError as e:
        print(f"\n⚠️  {e}")
        sys.exit(1)
    llm = LLMOrchestrator()
    print(f"\nModel : {llm.model}")
    print("\n[1] Health check ...")
    hc = llm.health_check()
    print("   ", _json.dumps(hc, indent=4))
    print("\n✅  Test complete")
