"""
agents/features/ritual_planner.py
────────────────────────────────────
Phase 5 — Scheduled Therapy Rituals Agent.

A real agent (not just a prompt method) that:
  - Learns user's response patterns to different ritual types
  - Schedules rituals based on session context and time of day
  - Tracks before/after mood deltas to measure ritual effectiveness
  - Adapts future recommendations based on what worked
  - Integrates with session_memory for persistent ritual history
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

RITUAL_TYPES = {
    "guided_breathing": {
        "name": "Guided Breathing",
        "duration_min": 5,
        "best_for": ["anxiety", "panic", "overwhelm", "pre-sleep"],
        "contraindicated": [],
    },
    "body_scan": {
        "name": "Body Scan",
        "duration_min": 10,
        "best_for": ["insomnia", "dissociation", "chronic pain", "low mood"],
        "contraindicated": ["acute_crisis"],
    },
    "journaling": {
        "name": "Guided Journaling",
        "duration_min": 15,
        "best_for": ["rumination", "cognitive_distortions", "processing", "goal_setting"],
        "contraindicated": [],
    },
    "gratitude": {
        "name": "Gratitude Practice",
        "duration_min": 5,
        "best_for": ["low_mood", "negativity_bias", "mild_depression"],
        "contraindicated": ["acute_crisis", "severe_depression"],
    },
    "grounding_54321": {
        "name": "5-4-3-2-1 Grounding",
        "duration_min": 5,
        "best_for": ["anxiety", "dissociation", "panic", "flashbacks"],
        "contraindicated": [],
    },
    "pmr": {
        "name": "Progressive Muscle Relaxation",
        "duration_min": 12,
        "best_for": ["physical_tension", "anxiety", "insomnia", "chronic_stress"],
        "contraindicated": [],
    },
    "values_reflection": {
        "name": "Values Reflection",
        "duration_min": 10,
        "best_for": ["meaning", "identity", "direction", "motivation"],
        "contraindicated": [],
    },
}


@dataclass
class RitualRecord:
    ritual_type:   str
    scheduled_at:  float
    completed_at:  Optional[float] = None
    mood_before:   Optional[int]   = None
    mood_after:    Optional[int]   = None
    mood_delta:    Optional[int]   = None  # after - before
    user_feedback: Optional[str]   = None
    completed:     bool            = False


@dataclass
class RitualRecommendation:
    ritual_type:    str
    ritual_name:    str
    duration_min:   int
    why:            str           # personalised reason
    steps:          List[str]
    duration_note:  str
    after_prompt:   str           # prompt to log mood after
    schedule_time:  Optional[str] = None  # suggested time (e.g. "tonight before bed")


# ── Ritual Planner Agent ──────────────────────────────────────────────────────

class RitualPlannerAgent:
    """
    Personalised therapy ritual planner.
    Learns from effectiveness data to improve future recommendations.
    """

    def __init__(self, llm, memory=None):
        self.llm    = llm
        self.memory = memory   # SessionMemory for persistent history
        self._ritual_history: Dict[str, List[RitualRecord]] = {}
        self._effectiveness: Dict[str, List[int]] = {}  # ritual_type → [mood_deltas]

    def recommend(
        self,
        session_id: str,
        mood_score:    int,
        user_context:  str,
        phq_severity:  Optional[str] = None,
        time_of_day:   Optional[str] = None,   # "morning" | "afternoon" | "evening" | "night"
    ) -> RitualRecommendation:
        """
        Recommend the best ritual for this user right now.
        Uses effectiveness history to personalise.
        """
        # Determine what's best based on context
        best_type = self._select_ritual_type(
            mood_score, user_context, phq_severity, time_of_day
        )
        ritual_meta = RITUAL_TYPES[best_type]

        # Build personalised ritual steps via LLM
        steps, why, after_prompt = self._generate_ritual_content(
            ritual_type=best_type,
            user_context=user_context,
            mood_score=mood_score,
        )

        # Schedule suggestion
        schedule_time = None
        if time_of_day in ("evening", "night"):
            schedule_time = "tonight before bed"
        elif time_of_day == "morning":
            schedule_time = "before you start your day"

        # Log in history
        record = RitualRecord(
            ritual_type=best_type,
            scheduled_at=time.time(),
            mood_before=mood_score,
        )
        if session_id not in self._ritual_history:
            self._ritual_history[session_id] = []
        self._ritual_history[session_id].append(record)

        return RitualRecommendation(
            ritual_type=best_type,
            ritual_name=ritual_meta["name"],
            duration_min=ritual_meta["duration_min"],
            why=why,
            steps=steps,
            duration_note=f"About {ritual_meta['duration_min']} minutes",
            after_prompt=after_prompt,
            schedule_time=schedule_time,
        )

    def log_completion(
        self,
        session_id: str,
        mood_after: int,
        feedback: Optional[str] = None,
    ) -> Dict:
        """
        Record that the user completed a ritual and log mood delta.
        Returns effectiveness summary.
        """
        history = self._ritual_history.get(session_id, [])
        # Find last incomplete ritual
        last_uncompleted = next(
            (r for r in reversed(history) if not r.completed), None
        )
        if last_uncompleted:
            last_uncompleted.completed    = True
            last_uncompleted.completed_at = time.time()
            last_uncompleted.mood_after   = mood_after
            last_uncompleted.user_feedback = feedback
            if last_uncompleted.mood_before:
                delta = mood_after - last_uncompleted.mood_before
                last_uncompleted.mood_delta = delta
                # Update effectiveness tracking
                rt = last_uncompleted.ritual_type
                if rt not in self._effectiveness:
                    self._effectiveness[rt] = []
                self._effectiveness[rt].append(delta)
                return {
                    "ritual_type": rt,
                    "mood_before": last_uncompleted.mood_before,
                    "mood_after":  mood_after,
                    "mood_delta":  delta,
                    "message":     self._effectiveness_message(delta),
                }
        return {"message": "No pending ritual found to complete."}

    def _effectiveness_message(self, delta: int) -> str:
        if delta >= 3:
            return "That ritual seems to be working really well for you."
        if delta >= 1:
            return "There was a gentle positive shift. Consistency helps."
        if delta == 0:
            return "No change today — that's okay. We can try a different approach next time."
        return "Mood dipped after the ritual — let's talk about what came up."

    def _select_ritual_type(
        self,
        mood_score: int,
        user_context: str,
        phq_severity: Optional[str],
        time_of_day: Optional[str],
    ) -> str:
        ctx_lower = user_context.lower()

        # Contraindication check
        if phq_severity in ("severe", "moderately severe"):
            excluded = {"gratitude"}
        else:
            excluded = set()

        # Context-based matching
        if any(w in ctx_lower for w in ["sleep", "insomnia", "can't sleep", "منام"]):
            if "body_scan" not in excluded:
                return "body_scan"
        if any(w in ctx_lower for w in ["panic", "anxious", "anxiety", "قلق"]):
            return "grounding_54321"
        if any(w in ctx_lower for w in ["ruminate", "overthink", "spiral", "thought"]):
            return "journaling"
        if any(w in ctx_lower for w in ["tense", "tight", "sore", "stress", "توتر"]):
            return "pmr"
        if time_of_day in ("evening", "night") and mood_score <= 5:
            return "body_scan"
        if mood_score <= 3 and "gratitude" not in excluded:
            return "guided_breathing"   # low mood → start gentle
        if any(w in ctx_lower for w in ["meaning", "purpose", "goal", "هدف"]):
            return "values_reflection"

        # Check effectiveness history
        best = self._get_most_effective_ritual(excluded)
        if best:
            return best

        # Default
        return "guided_breathing"

    def _get_most_effective_ritual(self, excluded: set) -> Optional[str]:
        if not self._effectiveness:
            return None
        scored = {
            rt: sum(deltas) / len(deltas)
            for rt, deltas in self._effectiveness.items()
            if rt not in excluded and len(deltas) >= 2
        }
        if not scored:
            return None
        return max(scored, key=scored.get)

    def _generate_ritual_content(
        self, ritual_type: str, user_context: str, mood_score: int
    ) -> Tuple[List[str], str, str]:
        """Generate personalised ritual steps, why, and after-prompt."""
        ritual_name = RITUAL_TYPES[ritual_type]["name"]
        system = (
            f"You are the MindBridge Ritual Planner. Generate a personalised {ritual_name} ritual.\n\n"
            f"User context: {user_context}\n"
            f"Current mood: {mood_score}/10\n\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"why": "one sentence linking to their specific situation", '
            '"steps": ["step 1 (warm, accessible)", "step 2", "step 3", "step 4"], '
            '"after_prompt": "prompt to log mood after completing"}'
        )
        raw = self.llm.generate(
            system_prompt=system,
            conversation=[{"role": "user", "content": f"Plan a {ritual_name} for me."}],
        )
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
            return data.get("steps", []), data.get("why", ""), data.get("after_prompt", "How do you feel now?")
        except Exception:
            return (
                ["Find a comfortable position.", "Close your eyes.", "Follow your breath."],
                f"Given what you've shared, a {ritual_name} may help.",
                "How do you feel after completing this ritual? (1-10)",
            )

    def get_effectiveness_summary(self, session_id: str) -> Dict:
        history = self._ritual_history.get(session_id, [])
        completed = [r for r in history if r.completed]
        if not completed:
            return {"message": "No completed rituals yet."}
        avg_delta = sum(r.mood_delta for r in completed if r.mood_delta is not None) / len(completed)
        by_type = {}
        for r in completed:
            if r.ritual_type not in by_type:
                by_type[r.ritual_type] = []
            if r.mood_delta is not None:
                by_type[r.ritual_type].append(r.mood_delta)
        return {
            "total_completed":  len(completed),
            "avg_mood_delta":   round(avg_delta, 2),
            "by_type":          {k: round(sum(v)/len(v), 2) for k, v in by_type.items() if v},
            "most_effective":   max(by_type, key=lambda k: sum(by_type[k])/len(by_type[k])) if by_type else None,
        }
