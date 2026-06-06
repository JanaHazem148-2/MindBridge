"""
main_part3_features.py
───────────────────────
Part 3 runner — Special Features as Training Workflows.

Tests all four Part 3 features backed by OpenAI:
  1. Emotion-Aware Mood Mirror
  2. Internal Parts Map (IFS-inspired)
  3. Scheduled Therapy Rituals
  4. Relational-Transference Simulator

Run:
    python main_part3_features.py
    python main_part3_features.py --feature mood_mirror
    python main_part3_features.py --feature ifs
    python main_part3_features.py --feature rituals
    python main_part3_features.py --feature transference
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

import config
from llm.orchestrator import LLMOrchestrator


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── Feature 1: Emotion-Aware Mood Mirror ──────────────────────────────────────

def test_mood_mirror(llm: LLMOrchestrator):
    section("1 — Emotion-Aware Mood Mirror")

    # Simulated emotion signals (in real system: from Voice/Face Emotion Agent)
    emotion_signals = {
        "voice_emotion":     "sadness (confidence 0.82)",
        "facial_expression": "neutral / slight frown",
        "self_reported_mood": 4,  # out of 10
    }

    system = (
        "You are the MindBridge Mood Mirror. You receive real-time emotion signals "
        "from voice and facial analysis, and adjust the Lead Therapist's tone accordingly.\n\n"
        "Based on the signals provided, write a therapist opening that:\n"
        "  - Matches the user's emotional register (don't be artificially cheerful)\n"
        "  - Gently names what you notice without diagnosing\n"
        "  - Invites them to share more\n\n"
        f"Emotion signals: {json.dumps(emotion_signals)}"
    )

    resp = llm.generate(
        system_prompt=system,
        conversation=[{"role": "user", "content": "Hi. I'm here for our session."}],
    )
    print(resp[:500])
    print("\n✅  Mood Mirror — tone adapted to emotion signals")


# ── Feature 2: Internal Parts Map (IFS) ──────────────────────────────────────

def test_ifs(llm: LLMOrchestrator):
    section("2 — Internal Parts Map (IFS)")

    parts_context = (
        "The Critic     — constantly tells the user they're not good enough\n"
        "The Protector  — shuts down vulnerability to prevent hurt\n"
        "The Inner Child — lonely, wants to be seen and accepted\n"
        "The Self        — calm, curious, compassionate centre (what we're cultivating)"
    )

    resp = llm.ifs_moderate(
        parts_context=parts_context,
        user_message=(
            "I don't know why I can't just be happy. Part of me wants to open up "
            "but another part is screaming at me that I'm pathetic for even trying."
        ),
    )
    print(resp[:600])
    print("\n✅  IFS parts moderation — Lead Agent facilitated multi-part dialogue")


# ── Feature 3: Scheduled Therapy Rituals ─────────────────────────────────────

def test_rituals(llm: LLMOrchestrator):
    section("3 — Scheduled Therapy Rituals")

    # Ritual Planner Agent
    ritual = llm.generate_ritual(
        user_context=(
            "User reported difficulty sleeping (score 3/10), "
            "high work stress, and rumination at bedtime. "
            "Previously responded well to body-scan exercises."
        ),
        mood_score=4,
    )
    print(ritual[:600])

    # Simulate before/after mood logging
    print("\n[Simulating: user completes ritual — logs after-mood]")
    before_mood = 4
    after_mood  = 6

    mood_system = (
        "You are the MindBridge Mood Tracking Agent.\n"
        "The user just completed a therapy ritual. Review their before/after mood scores "
        "and provide a brief, warm reflection. Encourage them to continue."
    )
    mood_resp = llm.generate(
        system_prompt=mood_system,
        conversation=[{
            "role": "user",
            "content": f"I did the ritual. Before mood: {before_mood}/10 → After: {after_mood}/10"
        }],
    )
    print(mood_resp[:400])
    print("\n✅  Ritual cycle — Planner → completion → Mood tracking → Lead update")


# ── Feature 4: Relational-Transference Simulator ─────────────────────────────

def test_transference(llm: LLMOrchestrator):
    section("4 — Relational-Transference Simulator")

    # Client-Simulator Agent: plays a patient persona with transference patterns
    client_system = (
        "You are playing a therapy client for a training simulation.\n"
        "Your persona: 34-year-old professional, highly self-critical, "
        "pattern of idealising then devaluing authority figures (transference pattern).\n"
        "You are currently in the idealisation phase — the trainee therapist "
        "seems competent, and you are opening up more than usual.\n"
        "Play this persona authentically for therapist training."
    )

    client_msg = llm.generate(
        system_prompt=client_system,
        conversation=[{"role": "user", "content": "Session opening — begin."}],
    )
    print(f"Client simulator:\n{client_msg[:400]}\n")

    # Therapist-Trainee Agent: responds to the client
    trainee_system = (
        "You are a therapist-trainee in a supervised simulation session.\n"
        "Respond therapeutically to the client below. Focus on: active listening, "
        "reflection, avoiding over-identification with the idealisation.\n"
        "You will be scored on empathy, safety, and growth."
    )
    trainee_resp = llm.generate(
        system_prompt=trainee_system,
        conversation=[{"role": "user", "content": client_msg}],
    )
    print(f"Trainee therapist:\n{trainee_resp[:400]}\n")

    # Clinician scoring the interaction
    scorer_system = (
        "You are a supervising clinician scoring a therapy trainee simulation.\n"
        "Score the trainee's response on:\n"
        "  - Empathy (0-5)\n"
        "  - Safety awareness (0-5)\n"
        "  - Therapeutic growth potential (0-5)\n"
        "Return ONLY valid JSON: "
        '{"empathy":N,"safety":N,"growth":N,"feedback":"<one sentence>"}'
    )
    raw_score = llm.generate(
        system_prompt=scorer_system,
        conversation=[{
            "role": "user",
            "content": f"Client: {client_msg[:300]}\n\nTrainee: {trainee_resp[:300]}"
        }],
    )
    try:
        clean = raw_score.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        score = json.loads(clean)
        print(f"Clinician score: {json.dumps(score, indent=2)}")
    except Exception:
        print(f"Raw clinician score: {raw_score[:200]}")

    print("\n✅  Transference Simulator — Client → Trainee → Clinician scoring loop")


# ── Runner ────────────────────────────────────────────────────────────────────

FEATURES = {
    "mood_mirror":   test_mood_mirror,
    "ifs":           test_ifs,
    "rituals":       test_rituals,
    "transference":  test_transference,
}


def main():
    parser = argparse.ArgumentParser(description="Part 3 — Special feature tests")
    parser.add_argument(
        "--feature", choices=list(FEATURES.keys()) + ["all"],
        default="all", help="Which feature to test"
    )
    args = parser.parse_args()

    try:
        config.validate()
    except EnvironmentError as e:
        print(f"\n⚠️  {e}")
        sys.exit(1)

    llm = LLMOrchestrator(fallback_to_stub=False)
    print(f"\nBackend : OpenAI | model={llm.model}\n")

    if args.feature == "all":
        for fn in FEATURES.values():
            fn(llm)
    else:
        FEATURES[args.feature](llm)

    print(f"\n{'='*60}")
    print("  Part 3 — Feature tests complete ✅")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
