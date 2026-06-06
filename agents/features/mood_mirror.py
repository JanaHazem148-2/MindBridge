"""
agents/features/mood_mirror.py
────────────────────────────────
Phase 5 — Emotion-Aware Mood Mirror Agent.

Roadmap: "Mood Mirror (voice + face) — additive, build after core agents stable."

This is a real agent, not just a system prompt.

What it does:
  1. Accepts multi-modal emotion signals (voice, face, self-report, text)
  2. Analyses emotional state holistically
  3. Dynamically adjusts the Lead Therapist's tone and pacing
  4. Detects incongruence (says "fine" but signals distress)
  5. Maintains emotion timeline across the session
  6. Feeds emotional context back to the AgentOrchestrator

Architecture:
  EmotionSignal (input) → MoodMirrorAgent → ToneAdjustment (output to therapist)

In full deployment:
  - Voice emotion: fed from ASR + prosody model (e.g. SpeechBrain)
  - Facial expression: fed from face analysis model (e.g. DeepFace / FER+)
  - Both are async streams — this agent subscribes to them

Without voice/face (text-only mode):
  - Falls back to text-based emotion inference via LLM
  - Still provides tone adjustment and incongruence detection

Usage:
    mirror = MoodMirrorAgent(llm=llm_orchestrator)

    signals = EmotionSignals(
        voice_emotion="sadness",
        voice_confidence=0.82,
        facial_expression="neutral",
        self_reported_mood=4,
        text_sentiment="negative",
    )

    adjustment = mirror.process(session_id, user_message, signals)
    # adjustment.modified_system_prompt → inject into therapist's next call
    # adjustment.tone_note             → brief note for therapist
    # adjustment.incongruence_flag     → True if signals contradict self-report
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class EmotionSignals:
    """
    Multi-modal emotion signals for one turn.
    All fields are optional — the agent gracefully handles missing modalities.
    """
    # Voice prosody (from ASR+emotion model)
    voice_emotion:      Optional[str]   = None   # "sadness" | "anger" | "fear" | "neutral" | "joy"
    voice_confidence:   float           = 0.0    # model confidence 0-1

    # Facial expression (from face analysis)
    facial_expression:  Optional[str]   = None   # "neutral" | "sad" | "anxious" | "angry" | "happy"
    facial_confidence:  float           = 0.0

    # Self-report (from mood_log tool or user input)
    self_reported_mood: Optional[int]   = None   # 1-10

    # Text-derived (from NLP sentiment)
    text_sentiment:     Optional[str]   = None   # "positive" | "neutral" | "negative" | "distressed"
    text_emotion_words: List[str]       = field(default_factory=list)  # ["hopeless", "tired"]

    # Derived by agent
    incongruence_detected: bool = False   # signals contradict each other
    primary_emotion:       str  = "unknown"


@dataclass
class ToneAdjustment:
    """
    Output of MoodMirrorAgent — instructions to the Lead Therapist.
    """
    session_id:            str
    primary_emotion:       str
    emotion_intensity:     str           # "low" | "medium" | "high" | "critical"
    incongruence_flag:     bool          # self-report ≠ non-verbal signals
    tone_instructions:     str           # injected into therapist system prompt
    opening_suggestion:    str           # suggested opening line for therapist
    pacing_note:           str           # "slow down" | "brief" | "normal" | "urgent"
    modified_system_prompt: str          # full context block for therapist
    flags:                 List[str]     # any clinical flags to log
    latency_ms:            float = 0.0


# ── Emotion analysis ──────────────────────────────────────────────────────────

EMOTION_TONE_MAP = {
    "sadness":     ("gentle, slow, unhurried",      "I can hear that something feels heavy today."),
    "anger":       ("calm, non-reactive, validating","It sounds like something has really frustrated you."),
    "fear":        ("grounding, steady, safe",       "I'm here with you. We can take this at whatever pace feels right."),
    "joy":         ("warm, matching energy",         "You seem lighter today — what's been going well?"),
    "neutral":     ("warm, curious, inviting",       "How are you coming in today?"),
    "distress":    ("slow, present, crisis-aware",   "Something in how you're sharing tells me today might be heavy. I'm here."),
    "unknown":     ("warm, attentive",               "I'm glad you're here today."),
}

INTENSITY_MAP = {
    # (mood_score_threshold, intensity)
    "critical": lambda m: m is not None and m <= 2,
    "high":     lambda m: m is not None and m <= 4,
    "medium":   lambda m: m is not None and m <= 6,
    "low":      lambda m: True,  # default
}


class MoodMirrorAgent:
    """
    Emotion-aware tone adjustment agent.
    Sits between the user's signals and the Lead Therapist's system prompt.
    """

    def __init__(self, llm=None, enable_llm_inference: bool = True):
        self.llm = llm
        self.enable_llm_inference = enable_llm_inference and (llm is not None)
        self._session_emotion_history: Dict[str, List[Dict]] = {}

    def process(
        self,
        session_id: str,
        user_message: str,
        signals: Optional[EmotionSignals] = None,
    ) -> ToneAdjustment:
        """
        Main entry point. Analyses signals and produces tone adjustment.
        """
        t_start = time.time()
        signals = signals or EmotionSignals()
        flags   = []

        # 1. Infer primary emotion
        primary = self._infer_primary_emotion(signals, user_message)
        signals.primary_emotion = primary

        # 2. Detect incongruence
        incongruence = self._detect_incongruence(signals)
        signals.incongruence_detected = incongruence
        if incongruence:
            flags.append("emotion_incongruence")

        # 3. Determine intensity
        intensity = self._determine_intensity(signals)

        # 4. Get tone instructions
        tone_style, opening = EMOTION_TONE_MAP.get(primary, EMOTION_TONE_MAP["unknown"])

        # 5. Pacing
        pacing = {
            "critical": "urgent",
            "high":     "slow down",
            "medium":   "normal",
            "low":      "normal",
        }[intensity]

        # 6. Build modified system prompt context block
        context_lines = [
            "[MOOD MIRROR — INTERNAL — do NOT share this with user]",
            f"Primary emotion detected: {primary} (intensity: {intensity})",
            f"Tone instructions: {tone_style}",
        ]
        if incongruence:
            context_lines.append(
                "⚠ INCONGRUENCE DETECTED: user's words and non-verbal signals don't match. "
                "Gently acknowledge what you notice without confronting. "
                "Example: 'You say you're okay, but I notice something in how you're sharing that. "
                "Is there something more you want to say?'"
            )
        if signals.self_reported_mood and signals.self_reported_mood <= 3:
            context_lines.append(
                f"Self-reported mood is very low ({signals.self_reported_mood}/10). "
                "Do NOT open with a task or structured exercise. Just be present."
            )
            flags.append("low_self_report")

        if intensity in ("critical", "high"):
            context_lines.append(
                "The user may be in significant distress. Slow your responses. "
                "Validate before anything else. "
                "Do not introduce CBT techniques until the user feels heard."
            )

        context_lines.append("[END MOOD MIRROR]")

        # 7. Update history
        self._update_history(session_id, primary, intensity, incongruence)

        # 8. LLM-enhanced tone adjustment (optional, for nuanced cases)
        if self.enable_llm_inference and intensity in ("high", "critical"):
            try:
                opening = self._llm_generate_opening(user_message, signals, primary, intensity)
            except Exception as e:
                logger.warning(f"LLM tone generation failed: {e} — using heuristic")

        return ToneAdjustment(
            session_id=session_id,
            primary_emotion=primary,
            emotion_intensity=intensity,
            incongruence_flag=incongruence,
            tone_instructions=tone_style,
            opening_suggestion=opening,
            pacing_note=pacing,
            modified_system_prompt="\n".join(context_lines),
            flags=flags,
            latency_ms=round((time.time() - t_start) * 1000, 1),
        )

    def _infer_primary_emotion(self, signals: EmotionSignals, text: str) -> str:
        """Infer primary emotion from available signals (priority order)."""
        # If both voice and face agree — high confidence
        if (signals.voice_emotion and signals.facial_expression and
                signals.voice_emotion.lower() in signals.facial_expression.lower()):
            return signals.voice_emotion.lower()

        # High-confidence voice
        if signals.voice_emotion and signals.voice_confidence >= 0.70:
            return signals.voice_emotion.lower()

        # Text emotion words (explicit distress language)
        distress_words = {"hopeless", "worthless", "can't go on", "no point",
                          "مفيش أمل", "تعبان", "مش قادر", "حاسس إني"}
        if any(w in text.lower() for w in distress_words):
            return "distress"

        # Text sentiment
        if signals.text_sentiment == "distressed":
            return "distress"
        if signals.text_sentiment == "negative":
            return "sadness"

        # Self-reported mood
        if signals.self_reported_mood is not None:
            if signals.self_reported_mood <= 3:
                return "sadness"
            if signals.self_reported_mood >= 8:
                return "joy"

        # Face only
        if signals.facial_expression and signals.facial_confidence >= 0.60:
            face_map = {"sad": "sadness", "angry": "anger", "fear": "fear",
                        "happy": "joy", "neutral": "neutral"}
            return face_map.get(signals.facial_expression.lower(), "unknown")

        return "neutral"

    def _detect_incongruence(self, signals: EmotionSignals) -> bool:
        """Detect when self-report contradicts non-verbal signals."""
        if signals.self_reported_mood and signals.self_reported_mood >= 7:
            # User says they're okay but signals say otherwise
            if signals.voice_emotion in ("sadness", "fear", "anger") and signals.voice_confidence >= 0.65:
                return True
            if signals.facial_expression in ("sad", "fear", "angry") and signals.facial_confidence >= 0.65:
                return True
        return False

    def _determine_intensity(self, signals: EmotionSignals) -> str:
        for level in ("critical", "high", "medium", "low"):
            if INTENSITY_MAP[level](signals.self_reported_mood):
                return level
        return "low"

    def _llm_generate_opening(
        self, user_message: str, signals: EmotionSignals, emotion: str, intensity: str
    ) -> str:
        system = (
            "You are the MindBridge Mood Mirror. Generate ONE opening sentence for a therapist "
            "responding to a distressed user. The sentence must:\n"
            "- Match their emotional register (not artificially positive)\n"
            "- Gently name what you notice without diagnosing\n"
            "- Invite them to share more\n"
            "- Be warm, calm, and 1-2 sentences maximum\n"
            f"Emotion detected: {emotion} (intensity: {intensity})\n"
            "Return ONLY the opening sentence, nothing else."
        )
        return self.llm.generate(
            system_prompt=system,
            conversation=[{"role": "user", "content": user_message}],
        )

    def _update_history(self, session_id: str, emotion: str, intensity: str, incongruence: bool):
        if session_id not in self._session_emotion_history:
            self._session_emotion_history[session_id] = []
        self._session_emotion_history[session_id].append({
            "emotion": emotion, "intensity": intensity,
            "incongruence": incongruence, "timestamp": time.time(),
        })
        # Keep last 20 turns
        self._session_emotion_history[session_id] = (
            self._session_emotion_history[session_id][-20:]
        )

    def get_emotion_timeline(self, session_id: str) -> List[Dict]:
        return self._session_emotion_history.get(session_id, [])

    def get_dominant_emotion(self, session_id: str) -> Optional[str]:
        history = self._session_emotion_history.get(session_id, [])
        if not history:
            return None
        from collections import Counter
        counts = Counter(h["emotion"] for h in history)
        return counts.most_common(1)[0][0]


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mirror = MoodMirrorAgent()

    print("=== MoodMirrorAgent — Smoke Test ===\n")
    test_cases = [
        ("Hi, I've been feeling really down.",
         EmotionSignals(voice_emotion="sadness", voice_confidence=0.82,
                        self_reported_mood=3, text_sentiment="negative")),
        ("I'm fine, really.",
         EmotionSignals(voice_emotion="sadness", voice_confidence=0.75,
                        self_reported_mood=8)),  # incongruence case
        ("عايز أتكلم. أنا مش كويس.",
         EmotionSignals(text_sentiment="distressed",
                        text_emotion_words=["مش كويس"])),
    ]
    for msg, signals in test_cases:
        adj = mirror.process("test-session", msg, signals)
        print(f"Message  : {msg}")
        print(f"Emotion  : {adj.primary_emotion} ({adj.emotion_intensity})")
        print(f"Incon    : {adj.incongruence_flag}")
        print(f"Pacing   : {adj.pacing_note}")
        print(f"Opening  : {adj.opening_suggestion}")
        print(f"Flags    : {adj.flags}")
        print()
    print("✅ MoodMirrorAgent smoke test complete")
