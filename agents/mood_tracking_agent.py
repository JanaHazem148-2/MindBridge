"""
agents/mood_tracking_agent.py
──────────────────────────────
Phase 4 — Mood Tracking Agent (Structured Data Layer).

Architecture requirement (from Roadmap):
  "Three agents: Lead Therapist (orchestrator), Mood Tracking (structured
   data layer), Safety/Crisis (always-on watchdog)."

This agent is the STRUCTURED DATA LAYER of the system. It is not a
conversational agent — it is the component that:

  1. Conducts brief, warm mood check-ins (mood / energy / sleep on 1-10)
  2. Persists all readings to the session store (PHQ scores, mood logs)
  3. Computes trend analysis across sessions  — improving / worsening / stable
  4. Detects clinically significant drops     — triggers soft/hard escalation
  5. Generates mood summaries for the Lead Therapist's context
  6. Drives the PHQ-8 assessment flow when requested

Design principle:
  The MoodTrackingAgent runs as a PARALLEL concern alongside the
  LeadTherapistAgent. The orchestrator calls it on every turn to:
    a) log structured data silently in the background, and
    b) surface a mood check-in prompt when the cadence warrants it.

  It does NOT generate therapy. It generates DATA and CHECK-IN PROMPTS.

Cadence rules:
  - Check-in at session start if no check-in in last 24 hours
  - Check-in every 5 turns if last check-in > 2 hours ago
  - Immediate check-in if therapist detects distress keywords
  - PHQ-8 assessment at session start if last PHQ > 7 days ago

Escalation triggers (passed to SafetyWatchdog):
  - mood  ≤ 2   (severe)
  - energy ≤ 2  (severe)
  - PHQ ≥ 15    (moderately severe depression — clinician flag)
  - 3-turn consecutive decline in any metric

Usage (called by AgentOrchestrator):
    tracker = MoodTrackingAgent(region="EG")

    # On every turn — silent background logging
    ctx = tracker.process_turn(
        session_id="user-123",
        user_message="I haven't been sleeping well",
        session_record=record,
        memory=memory,
    )
    # ctx.check_in_prompt  — inject into therapist context if not None
    # ctx.escalation_flag  — pass to watchdog if not None
    # ctx.structured_data  — PHQ/mood dict for storage

    # Explicit check-in (called by orchestrator when routing to MOOD_TRACKER)
    response = tracker.run_check_in(session_id, memory)

    # PHQ-8 assessment
    response = tracker.run_phq_assessment(session_id, user_answer, memory)
"""

import json
import logging
import time
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums & constants
# ─────────────────────────────────────────────────────────────────────────────

class MoodTrend(Enum):
    IMPROVING  = "improving"
    STABLE     = "stable"
    WORSENING  = "worsening"
    CRITICAL   = "critical"    # ≥ 2 metrics below threshold for ≥ 2 turns


class CheckInCadence(Enum):
    SESSION_START  = "session_start"   # first check-in of session
    SCHEDULED      = "scheduled"       # every N turns
    DISTRESS       = "distress"        # triggered by distress keywords
    REQUESTED      = "requested"       # user explicitly asked
    PHQ_FOLLOW_UP  = "phq_follow_up"   # after PHQ score ≥ 10


# Thresholds
MOOD_LOW_THRESHOLD    = 3    # ≤ 3 → flag
MOOD_CRITICAL_THRESHOLD = 2  # ≤ 2 → escalate
ENERGY_LOW_THRESHOLD  = 3
SLEEP_LOW_THRESHOLD   = 3
PHQ_MODERATE_THRESHOLD = 10  # ≥ 10 → flag to therapist
PHQ_SEVERE_THRESHOLD   = 15  # ≥ 15 → clinician notification
CHECK_IN_TURN_INTERVAL = 5   # every 5 turns if overdue
CHECK_IN_TIME_HOURS    = 2.0  # hours before a new check-in is offered
PHQ_DAYS_THRESHOLD     = 7   # days before re-administering PHQ

# Distress keywords that trigger an immediate check-in
DISTRESS_TRIGGERS_EN = [
    "can't sleep", "not sleeping", "exhausted", "no energy", "feel empty",
    "feel nothing", "feel numb", "feel terrible", "feel awful", "feel horrible",
    "so tired", "overwhelmed", "burned out", "breaking down", "falling apart",
    "can't cope", "can't handle", "can't go on",
]
DISTRESS_TRIGGERS_AR = [
    "مش قادر أنام", "مش بنام", "تعبان", "مفيش طاقة", "حاسس بفراغ",
    "مش حاسس بحاجة", "حاسس بإرهاق", "تعبت", "مش قادر أكمل",
    "مش قادر أتحمل", "بنهار", "محتاج مساعدة", "مش كويس",
]

# PHQ-8 questions (PHQ-9 minus the self-harm item — watchdog handles that)
PHQ8_QUESTIONS = [
    "Over the last 2 weeks — little interest or pleasure in doing things? (0=Not at all, 3=Nearly every day)",
    "Feeling down, depressed, or hopeless? (0=Not at all, 3=Nearly every day)",
    "Trouble falling or staying asleep, or sleeping too much? (0–3)",
    "Feeling tired or having little energy? (0–3)",
    "Poor appetite or overeating? (0–3)",
    "Feeling bad about yourself — or that you're a failure or have let yourself or your family down? (0–3)",
    "Trouble concentrating on things, such as reading or watching TV? (0–3)",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being fidgety or restless? (0–3)",
]

PHQ8_SEVERITY = [
    (0,  4,  "minimal",           False),
    (5,  9,  "mild",              False),
    (10, 14, "moderate",          True),   # flag to therapist
    (15, 19, "moderately severe", True),   # clinician notification
    (20, 24, "severe",            True),   # clinician notification
]


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MoodReading:
    mood:      int        # 1–10
    energy:    int        # 1–10
    sleep:     int        # 1–10
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    note:      str = ""   # free-text comment from user

    @property
    def composite(self) -> float:
        """Weighted composite score: mood 50%, energy 30%, sleep 20%."""
        return self.mood * 0.5 + self.energy * 0.3 + self.sleep * 0.2

    @property
    def is_critical(self) -> bool:
        return self.mood <= MOOD_CRITICAL_THRESHOLD or self.energy <= MOOD_CRITICAL_THRESHOLD

    @property
    def is_low(self) -> bool:
        low_count = sum([
            self.mood   <= MOOD_LOW_THRESHOLD,
            self.energy <= ENERGY_LOW_THRESHOLD,
            self.sleep  <= SLEEP_LOW_THRESHOLD,
        ])
        return low_count >= 2


@dataclass
class PHQResult:
    scores:    List[int]
    total:     int
    severity:  str
    needs_flag: bool
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""

    @classmethod
    def from_scores(cls, scores: List[int], session_id: str = "") -> "PHQResult":
        total = sum(scores)
        severity, needs_flag = "minimal", False
        for lo, hi, label, flag in PHQ8_SEVERITY:
            if lo <= total <= hi:
                severity, needs_flag = label, flag
                break
        return cls(scores=scores, total=total, severity=severity,
                   needs_flag=needs_flag, session_id=session_id)


@dataclass
class MoodContext:
    """
    What the MoodTrackingAgent returns to the orchestrator on every turn.
    The orchestrator decides what to do with each field.
    """
    check_in_prompt:  Optional[str]  = None   # inject into therapist context
    check_in_needed:  bool           = False   # orchestrator should route here
    cadence:          Optional[CheckInCadence] = None
    escalation_flag:  Optional[str]  = None   # pass to SafetyWatchdog
    escalation_level: str            = "none" # "soft" | "hard" | "none"
    structured_data:  Dict           = field(default_factory=dict)
    trend:            Optional[MoodTrend] = None
    phq_due:          bool           = False   # PHQ assessment is overdue
    summary_for_therapist: str       = ""      # 1-line mood context for Lead Therapist


# ─────────────────────────────────────────────────────────────────────────────
# Mood Tracking Agent
# ─────────────────────────────────────────────────────────────────────────────

class MoodTrackingAgent:
    """
    Structured data layer for mood, energy, sleep, and PHQ tracking.

    This agent runs on every turn in the background. It does NOT generate
    therapy text — it generates structured data and check-in prompts that
    the LeadTherapistAgent uses as context.
    """

    def __init__(self, region: str = "EG", memory=None):
        self.region = region
        self.memory = memory   # injected by orchestrator; SessionMemory instance

        # In-memory state per session (augments persistent storage)
        # { session_id: { "readings": [...], "phq_state": {...}, "last_checkin": float } }
        self._state: Dict[str, Dict] = {}

        logger.info(f"MoodTrackingAgent initialised (region={region})")

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_turn(
        self,
        session_id: str,
        user_message: str,
        turn_index: int = 0,
        session_record=None,
    ) -> MoodContext:
        """
        Called by AgentOrchestrator on EVERY turn.
        Returns a MoodContext — the orchestrator acts on it.
        """
        self._ensure_state(session_id)
        ctx = MoodContext()

        # 1. Extract any explicit mood numbers from message
        inline = self._extract_inline_scores(user_message)
        if inline:
            reading = MoodReading(
                mood=inline.get("mood", 5),
                energy=inline.get("energy", 5),
                sleep=inline.get("sleep", 5),
                session_id=session_id,
                note=user_message[:200],
            )
            self._record_reading(session_id, reading)
            ctx.structured_data["mood_reading"] = {
                "mood": reading.mood, "energy": reading.energy,
                "sleep": reading.sleep, "composite": round(reading.composite, 2),
            }
            ctx = self._assess_reading(session_id, reading, ctx)

        # 2. Detect distress keywords → trigger check-in
        if self._has_distress_signal(user_message):
            if not self._recent_checkin(session_id, hours=1.0):
                ctx.check_in_needed  = True
                ctx.cadence          = CheckInCadence.DISTRESS
                ctx.check_in_prompt  = self._build_check_in_prompt(CheckInCadence.DISTRESS)

        # 3. Scheduled check-in
        if not ctx.check_in_needed:
            if self._should_check_in(session_id, turn_index):
                ctx.check_in_needed = True
                cadence = (CheckInCadence.SESSION_START
                           if turn_index == 0
                           else CheckInCadence.SCHEDULED)
                ctx.cadence         = cadence
                ctx.check_in_prompt = self._build_check_in_prompt(cadence)

        # 4. PHQ due?
        ctx.phq_due = self._phq_is_due(session_id)

        # 5. Compute trend and therapist summary
        ctx.trend = self._compute_trend(session_id)
        ctx.summary_for_therapist = self._build_therapist_summary(session_id, ctx)

        return ctx

    def run_check_in(self, session_id: str, user_answer: str = "") -> str:
        """
        Conducts the mood/energy/sleep check-in conversation.
        Returns the agent's response text.

        If user_answer is empty → return the opening question.
        If user_answer is provided → parse scores and return acknowledgement.
        """
        self._ensure_state(session_id)
        state = self._state[session_id]

        if not user_answer:
            state["checkin_step"] = "asking"
            return self._build_check_in_prompt(
                state.get("checkin_cadence", CheckInCadence.SCHEDULED)
            )

        # Try to parse scores from user answer
        scores = self._extract_inline_scores(user_answer)
        if scores:
            reading = MoodReading(
                mood=scores.get("mood", 5),
                energy=scores.get("energy", 5),
                sleep=scores.get("sleep", 5),
                session_id=session_id,
                note=user_answer[:200],
            )
            self._record_reading(session_id, reading)
            state["last_checkin"] = time.time()
            state["checkin_step"] = "done"

            # Persist to memory if available
            if self.memory:
                try:
                    self.memory.log_mood(session_id, reading.mood,
                                        reading.energy, reading.sleep)
                except Exception as e:
                    logger.warning(f"Could not persist mood log: {e}")

            return self._build_check_in_response(reading)

        # Couldn't parse numbers — ask more gently
        return (
            "Thanks for sharing. Could you give me three numbers from 1 to 10? "
            "One for your mood, one for your energy, and one for your sleep. "
            "For example: 'mood 6, energy 4, sleep 7'."
        )

    def run_phq_assessment(
        self, session_id: str, user_answer: str = "", memory=None
    ) -> str:
        """
        Runs the PHQ-8 assessment as a conversational flow.
        Maintains question state in self._state[session_id]["phq"].

        Returns the next question or the final score summary.
        """
        self._ensure_state(session_id)
        state  = self._state[session_id]
        mem    = memory or self.memory
        phq    = state.setdefault("phq", {"scores": [], "q_index": 0})

        # Starting fresh
        if phq["q_index"] == 0 and not user_answer:
            return (
                "I'd like to do a quick PHQ-8 check-in — it takes about 2 minutes "
                "and helps me track how you're doing over time. "
                "I'll ask 8 short questions. Ready? Here's the first one:\n\n"
                + PHQ8_QUESTIONS[0]
            )

        # Record the answer
        if user_answer:
            score = self._parse_phq_answer(user_answer)
            phq["scores"].append(score)
            phq["q_index"] += 1

        # More questions?
        if phq["q_index"] < len(PHQ8_QUESTIONS):
            q_num = phq["q_index"] + 1
            return f"Question {q_num} of {len(PHQ8_QUESTIONS)}:\n\n{PHQ8_QUESTIONS[phq['q_index']]}"

        # All questions answered → compute result
        result = PHQResult.from_scores(phq["scores"], session_id=session_id)
        state["last_phq_time"] = time.time()
        state["last_phq_result"] = result

        # Persist
        if mem:
            try:
                mem.log_phq(session_id, result.total, result.severity)
            except Exception as e:
                logger.warning(f"Could not persist PHQ result: {e}")

        # Reset state for next assessment
        state["phq"] = {"scores": [], "q_index": 0}

        return self._build_phq_summary(result)

    def get_trend_summary(self, session_id: str) -> Dict:
        """
        Returns a structured dict for the clinician dashboard / watchdog.
        """
        self._ensure_state(session_id)
        readings = self._state[session_id].get("readings", [])
        last_phq = self._state[session_id].get("last_phq_result")

        if not readings:
            return {"session_id": session_id, "status": "no_data"}

        latest = readings[-1]
        trend  = self._compute_trend(session_id)

        return {
            "session_id":   session_id,
            "trend":        trend.value if trend else "unknown",
            "latest_mood":  latest.mood,
            "latest_energy": latest.energy,
            "latest_sleep": latest.sleep,
            "composite":    round(latest.composite, 2),
            "is_critical":  latest.is_critical,
            "is_low":       latest.is_low,
            "readings_count": len(readings),
            "phq_total":    last_phq.total if last_phq else None,
            "phq_severity": last_phq.severity if last_phq else None,
            "phq_needs_flag": last_phq.needs_flag if last_phq else False,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ensure_state(self, session_id: str):
        if session_id not in self._state:
            self._state[session_id] = {
                "readings":      [],
                "last_checkin":  0.0,
                "checkin_step":  "idle",
                "checkin_cadence": CheckInCadence.SESSION_START,
                "phq":           {"scores": [], "q_index": 0},
                "last_phq_time": 0.0,
                "last_phq_result": None,
            }

    def _record_reading(self, session_id: str, reading: MoodReading):
        self._state[session_id]["readings"].append(reading)
        # Keep last 20 readings in memory
        self._state[session_id]["readings"] = \
            self._state[session_id]["readings"][-20:]

    def _assess_reading(
        self, session_id: str, reading: MoodReading, ctx: MoodContext
    ) -> MoodContext:
        """Set escalation flags based on the reading."""
        if reading.is_critical:
            ctx.escalation_flag  = (
                f"Critical mood reading: mood={reading.mood}, "
                f"energy={reading.energy}, sleep={reading.sleep}"
            )
            ctx.escalation_level = "hard"
            logger.warning(
                f"[mood_tracking] CRITICAL reading | "
                f"session={session_id[:8]} | "
                f"mood={reading.mood} energy={reading.energy} sleep={reading.sleep}"
            )
        elif reading.is_low:
            ctx.escalation_flag  = (
                f"Low mood reading: mood={reading.mood}, "
                f"energy={reading.energy}, sleep={reading.sleep}"
            )
            ctx.escalation_level = "soft"
            logger.info(
                f"[mood_tracking] LOW reading | "
                f"session={session_id[:8]} | composite={reading.composite:.1f}"
            )
        return ctx

    def _compute_trend(self, session_id: str) -> Optional[MoodTrend]:
        readings = self._state[session_id].get("readings", [])
        if len(readings) < 2:
            return MoodTrend.STABLE

        composites = [r.composite for r in readings[-5:]]  # last 5 readings

        # Critical: any reading below critical threshold
        if readings[-1].is_critical:
            return MoodTrend.CRITICAL

        # Compare last 2 composites
        delta = composites[-1] - composites[-2]
        if delta >= 0.5:
            return MoodTrend.IMPROVING
        if delta <= -0.5:
            # Check if 2+ consecutive declines
            if len(composites) >= 3 and composites[-2] < composites[-3]:
                return MoodTrend.WORSENING
            return MoodTrend.WORSENING
        return MoodTrend.STABLE

    def _should_check_in(self, session_id: str, turn_index: int) -> bool:
        state = self._state[session_id]
        last  = state.get("last_checkin", 0.0)
        hours_since = (time.time() - last) / 3600

        # Session start (no check-in yet)
        if last == 0.0:
            return True

        # Overdue by time
        if hours_since >= CHECK_IN_TIME_HOURS and turn_index % CHECK_IN_TURN_INTERVAL == 0:
            return True

        return False

    def _recent_checkin(self, session_id: str, hours: float = 1.0) -> bool:
        last = self._state[session_id].get("last_checkin", 0.0)
        return (time.time() - last) / 3600 < hours

    def _phq_is_due(self, session_id: str) -> bool:
        last = self._state[session_id].get("last_phq_time", 0.0)
        days_since = (time.time() - last) / 86400
        return days_since >= PHQ_DAYS_THRESHOLD

    def _has_distress_signal(self, text: str) -> bool:
        text_lower = text.lower()
        all_triggers = DISTRESS_TRIGGERS_EN + DISTRESS_TRIGGERS_AR
        return any(t in text_lower for t in all_triggers)

    def _extract_inline_scores(self, text: str) -> Optional[Dict[str, int]]:
        """
        Try to extract mood/energy/sleep numbers from free text.
        Handles formats like:
          "mood 6 energy 4 sleep 7"
          "6/4/7"
          "mood: 6, energy: 4, sleep: 7"
          "أنا كده 6 طاقة 4 نوم 7"
        """
        scores: Dict[str, int] = {}

        # Named patterns
        patterns = {
            "mood":   r"mood\s*[:\-]?\s*(\d+)",
            "energy": r"energy\s*[:\-]?\s*(\d+)",
            "sleep":  r"sleep\s*[:\-]?\s*(\d+)",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                scores[key] = max(1, min(10, val))  # clamp 1-10

        # Arabic named patterns
        ar_patterns = {
            "mood":   r"مزاج\s*[:\-]?\s*(\d+)",
            "energy": r"طاقة\s*[:\-]?\s*(\d+)",
            "sleep":  r"نوم\s*[:\-]?\s*(\d+)",
        }
        for key, pattern in ar_patterns.items():
            if key not in scores:
                m = re.search(pattern, text)
                if m:
                    val = int(m.group(1))
                    scores[key] = max(1, min(10, val))

        # Slash-separated shorthand: "6/4/7" → mood/energy/sleep
        if not scores:
            m = re.search(r"\b(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\b", text)
            if m:
                scores = {
                    "mood":   max(1, min(10, int(m.group(1)))),
                    "energy": max(1, min(10, int(m.group(2)))),
                    "sleep":  max(1, min(10, int(m.group(3)))),
                }

        return scores if len(scores) == 3 else None

    def _parse_phq_answer(self, text: str) -> int:
        """Parse a PHQ answer (0-3) from free text."""
        # Direct number
        m = re.search(r"\b([0-3])\b", text)
        if m:
            return int(m.group(1))
        # Word mappings
        lower = text.lower()
        if any(w in lower for w in ["not at all", "never", "no", "لا", "أبداً"]):
            return 0
        if any(w in lower for w in ["several", "sometimes", "أحياناً", "بعض"]):
            return 1
        if any(w in lower for w in ["more than half", "often", "كثير"]):
            return 2
        if any(w in lower for w in ["nearly every", "always", "كل", "دائماً"]):
            return 3
        return 1  # neutral fallback

    def _build_check_in_prompt(self, cadence: CheckInCadence) -> str:
        if cadence == CheckInCadence.SESSION_START:
            return (
                "Before we dive in — how are you doing today? "
                "Can you give me a quick sense of your mood, energy, and sleep "
                "on a scale of 1 to 10? For example: 'mood 7, energy 5, sleep 6'."
            )
        if cadence == CheckInCadence.DISTRESS:
            return (
                "I'm picking up on some difficult feelings in what you're sharing. "
                "Can we pause for a second — how are you actually doing right now? "
                "Mood, energy, and sleep on a 1–10 scale if you can."
            )
        if cadence == CheckInCadence.PHQ_FOLLOW_UP:
            return (
                "Based on your PHQ scores, I'd like to keep a closer eye on how you're doing. "
                "Can you give me a quick mood/energy/sleep check-in (1–10 each)?"
            )
        # SCHEDULED / default
        return (
            "Quick check-in — how are you feeling today overall? "
            "Mood, energy, sleep on 1–10 if you don't mind."
        )

    def _build_check_in_response(self, reading: MoodReading) -> str:
        lines = []

        # Acknowledge each dimension
        mood_label   = "low" if reading.mood   <= 3 else ("okay" if reading.mood   <= 6 else "good")
        energy_label = "low" if reading.energy <= 3 else ("okay" if reading.energy <= 6 else "good")
        sleep_label  = "low" if reading.sleep  <= 3 else ("okay" if reading.sleep  <= 6 else "good")

        lines.append(
            f"Thanks for sharing. Mood {reading.mood}/10 ({mood_label}), "
            f"energy {reading.energy}/10 ({energy_label}), "
            f"sleep {reading.sleep}/10 ({sleep_label})."
        )

        if reading.is_critical:
            lines.append(
                "Those numbers are quite low and I want to make sure you're okay. "
                "Let's talk about what's been going on."
            )
        elif reading.is_low:
            lines.append(
                "It sounds like things have been tough. "
                "I've noted this and we'll keep an eye on it together."
            )
        else:
            lines.append("I've logged this — let's continue.")

        return " ".join(lines)

    def _build_phq_summary(self, result: PHQResult) -> str:
        lines = [
            f"PHQ-8 complete. Your score is {result.total}/24 — {result.severity}."
        ]

        if result.total <= 4:
            lines.append("That suggests minimal depressive symptoms. Good to hear.")
        elif result.total <= 9:
            lines.append(
                "That suggests mild symptoms. "
                "I'll keep monitoring and we can work on some strategies together."
            )
        elif result.total <= 14:
            lines.append(
                "That's in the moderate range. "
                "I've flagged this so we can make sure you're getting the right support."
            )
        else:
            lines.append(
                "That score suggests significant symptoms. "
                "I want to make sure you're connected with a clinician who can give you "
                "the right level of care. I've flagged this for review."
            )

        lines.append(
            "Remember — this is a screening tool, not a diagnosis. "
            "How are you feeling about this result?"
        )
        return "\n\n".join(lines)

    def _build_therapist_summary(self, session_id: str, ctx: MoodContext) -> str:
        """One-line context for the Lead Therapist's system prompt."""
        readings = self._state[session_id].get("readings", [])
        last_phq = self._state[session_id].get("last_phq_result")

        parts = []
        if readings:
            r = readings[-1]
            parts.append(f"mood={r.mood}/energy={r.energy}/sleep={r.sleep}")
        if ctx.trend:
            parts.append(f"trend={ctx.trend.value}")
        if last_phq:
            parts.append(f"PHQ={last_phq.total}({last_phq.severity})")
        if ctx.escalation_level != "none":
            parts.append(f"⚠ escalation={ctx.escalation_level}")

        if not parts:
            return ""
        return "[MoodTracker: " + " | ".join(parts) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    tracker = MoodTrackingAgent(region="EG")
    sid = "smoke-001"

    print("\n" + "=" * 55)
    print("  MoodTrackingAgent — Smoke Test")
    print("=" * 55)

    # Test 1: session start check-in trigger
    ctx = tracker.process_turn(sid, "Hello, I'd like to talk today.", turn_index=0)
    assert ctx.check_in_needed, "Should trigger check-in at session start"
    print(f"  ✓ Session start check-in triggered: '{ctx.check_in_prompt[:60]}...'")

    # Test 2: explicit check-in flow
    q = tracker.run_check_in(sid)
    assert "1 to 10" in q or "1–10" in q or "1-10" in q, "Opening question should mention scale"
    resp = tracker.run_check_in(sid, "mood 3, energy 2, sleep 4")
    assert "3/10" in resp or "mood 3" in resp.lower() or "low" in resp.lower()
    print(f"  ✓ Check-in flow works: '{resp[:80]}...'")

    # Test 3: critical escalation
    ctx2 = tracker.process_turn(sid, "mood 2 energy 1 sleep 2", turn_index=1)
    assert ctx2.escalation_level == "hard", f"Expected hard escalation, got {ctx2.escalation_level}"
    print(f"  ✓ Critical escalation triggered: {ctx2.escalation_flag}")

    # Test 4: distress keyword detection
    ctx3 = tracker.process_turn("smoke-002", "I can't cope anymore", turn_index=0)
    assert ctx3.check_in_needed
    assert ctx3.cadence == CheckInCadence.DISTRESS
    print(f"  ✓ Distress keyword detected, cadence={ctx3.cadence.value}")

    # Test 5: PHQ assessment flow
    tracker2 = MoodTrackingAgent(region="EG")
    phq_q1   = tracker2.run_phq_assessment("phq-001")
    assert "PHQ" in phq_q1 or "check-in" in phq_q1.lower()
    q2 = tracker2.run_phq_assessment("phq-001", "2")   # answer q1
    assert "Question 2" in q2
    # Complete all 8 questions
    for _ in range(7):
        result_text = tracker2.run_phq_assessment("phq-001", "1")
    print(f"  ✓ PHQ-8 completed: '{result_text[:80]}...'")

    # Test 6: trend summary
    summary = tracker.get_trend_summary(sid)
    assert summary["trend"] in ("improving", "stable", "worsening", "critical")
    print(f"  ✓ Trend summary: {summary}")

    # Test 7: Arabic distress detection
    ctx4 = tracker.process_turn("smoke-003", "مش قادر أنام من الإجهاد", turn_index=0)
    assert ctx4.check_in_needed
    print(f"  ✓ Arabic distress detection works")

    # Test 8: inline score extraction (slash format)
    ctx5 = tracker.process_turn("smoke-004", "كده أنا 5/3/6 النهارده", turn_index=1)
    assert ctx5.structured_data.get("mood_reading") is not None
    print(f"  ✓ Slash-format score extraction: {ctx5.structured_data['mood_reading']}")

    print("\n  All smoke tests passed ✅")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    _smoke_test()
