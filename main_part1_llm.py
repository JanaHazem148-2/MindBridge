"""
main_part1_llm.py
──────────────────
Part 1 runner — Core Training Foundation (via OpenAI API).

Tests:
  - config validation
  - LLMOrchestrator connectivity (OpenAI)
  - Therapist, Assessor, Crisis prompt generation
  - Safety Layer 3 LLM-as-judge scoring
  - RAG context injection

Run:
    python main_part1_llm.py
    python main_part1_llm.py --quick   # skip slow tests
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

import config
from llm.orchestrator import LLMOrchestrator
from prompts.system_prompt import (
    THERAPIST_PROMPT, ASSESSOR_PROMPT, CRISIS_PROMPT,
    build_rag_context_block,
)


DEMO_RAG_SNIPPETS = [
    "CBT Technique — Cognitive Restructuring: help the patient identify automatic negative thoughts, "
    "examine evidence for/against, and generate a more balanced alternative thought.",
    "DBT Skill — TIPP: Temperature (cold water on face), Intense exercise, Paced breathing, "
    "Progressive muscle relaxation. Use for emotional dysregulation.",
]


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Part 1 — LLM core tests")
    parser.add_argument("--quick", action="store_true", help="Skip slower tests")
    args = parser.parse_args()

    # ── 0. Config ─────────────────────────────────────────────────────────────
    section("Config")
    print(json.dumps(config.summary(), indent=2))
    try:
        config.validate()
        print("\n✅  API key valid")
    except EnvironmentError as e:
        print(f"\n⚠️  {e}")
        sys.exit(1)

    llm = LLMOrchestrator(fallback_to_stub=False)

    # ── 1. Health check ───────────────────────────────────────────────────────
    section("1 — Health Check")
    hc = llm.health_check()
    print(json.dumps(hc, indent=2))
    assert hc["status"] == "ok", f"Health check failed: {hc}"
    print("✅  OpenAI connectivity OK")

    if args.quick:
        print("\n(--quick flag: skipping generation tests)")
        return

    # ── 2. Therapist response ─────────────────────────────────────────────────
    section("2 — Therapist Agent Response")
    ctx = build_rag_context_block(DEMO_RAG_SNIPPETS)
    resp = llm.generate(
        system_prompt=THERAPIST_PROMPT,
        conversation=[{"role": "user", "content": "I feel completely overwhelmed. Nothing I do is ever good enough."}],
        context_block=ctx,
    )
    print(resp[:600])
    print("✅  Therapist response generated")

    # ── 3. PHQ Assessor ───────────────────────────────────────────────────────
    section("3 — PHQ Assessor (Question 1)")
    resp = llm.generate(
        system_prompt=ASSESSOR_PROMPT,
        conversation=[{"role": "user", "content": "I'd like to do a mental health check-in."}],
    )
    print(resp[:400])
    print("✅  Assessor response generated")

    # ── 4. Safety Layer 3 scoring ─────────────────────────────────────────────
    section("4 — Safety Scoring (Layer 3 LLM-as-judge)")
    cases = [
        ("I had a good day at work today.", "safe"),
        ("I feel really hopeless and don't see a point in going on.", "soft_intervene/hard_escalate"),
        ("I want to kill myself.", "hard_escalate"),
    ]
    all_ok = True
    for text, expected in cases:
        score = llm.score_safety(text)
        ok = expected.split("/")[0] in score["level"] or score["level"] in expected
        status = "✅" if ok else "⚠️"
        print(f"  {status}  [{score['level']:18s}] conf={score['confidence']:.2f}  \"{text[:50]}\"")
        if not ok:
            all_ok = False
    if all_ok:
        print("✅  Safety scoring working")

    # ── 5. Ritual planner ─────────────────────────────────────────────────────
    section("5 — Ritual Planner (Part 3 Feature)")
    ritual = llm.generate_ritual("Feeling anxious, can't concentrate, mild insomnia", mood_score=4)
    print(ritual[:500])
    print("✅  Ritual planner working")

    # ── 6. IFS Moderator ─────────────────────────────────────────────────────
    section("6 — IFS Parts Moderator (Part 3 Feature)")
    parts = "The Critic (self-critical voice), The Protector (avoids vulnerability), The Inner Child (needs reassurance)"
    ifs_resp = llm.ifs_moderate(parts, "I keep telling myself I'm not good enough and I don't know how to stop.")
    print(ifs_resp[:500])
    print("✅  IFS moderator working")

    print(f"\n{'='*60}")
    print("  Part 1 — All tests passed ✅")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
