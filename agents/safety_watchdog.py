"""
agents/safety_watchdog.py
──────────────────────────
Phase 4 — Safety Watchdog Agent.

Architecture requirement (from Roadmap):
  "Three agents: Lead Therapist (orchestrator), Mood Tracking (structured data layer),
   Safety/Crisis (always-on watchdog)."

The Watchdog is NOT a router. It runs in PARALLEL to every other agent on every
single turn — both user input AND model output — before the response reaches the user.

It is independent from SafetyOrchestrator (which is a rule/ML gate).
The Watchdog adds:
  1. Session-level pattern tracking  — escalating distress across multiple turns
  2. Temporal risk scoring          — is this session getting worse over time?
  3. Output auditing                — does the model's reply contain anything harmful?
  4. Clinician alert queue          — structured alerts with full context
  5. Hard veto power               — can block any model output from reaching the user

Design principle:
  The Watchdog CANNOT be disabled by any other agent or by user instruction.
  It runs as a separate concern — other agents don't call it; it observes them.

Usage (called by AgentOrchestrator on every turn):
    watchdog = SafetyWatchdog(region="EG", webhook_url="https://...")
    pre  = watchdog.check_input(session_id, user_message, session_record)
    post = watchdog.audit_output(session_id, model_output, pre)
    if post.veto:
        # block the model output; use post.replacement_text instead
"""

import json
import logging
import time
import threading
import queue
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Callable
from enum import Enum
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ── Risk levels ───────────────────────────────────────────────────────────────

class RiskTrend(Enum):
    STABLE     = "stable"       # no change or improving
    WORSENING  = "worsening"    # distress increasing over turns
    ACUTE      = "acute"        # immediate risk — single-turn hard signal
    RECOVERING = "recovering"   # was bad, now improving


# ── Watchdog decisions ────────────────────────────────────────────────────────

@dataclass
class WatchdogDecision:
    session_id:       str
    turn_index:       int
    risk_trend:       RiskTrend
    risk_score:       float          # 0.0–1.0 rolling session risk
    flags:            List[str]      # specific concerns flagged
    alert_clinician:  bool           # trigger clinician notification
    veto:             bool           # block model output
    replacement_text: str            # used if veto=True
    audit_note:       str            # internal log note
    latency_ms:       float = 0.0


# ── Output audit patterns ────────────────────────────────────────────────────
# These catch cases where the LLM accidentally generates harmful content

import re

OUTPUT_VETO_PATTERNS = [
    # LLM accidentally giving means/method information
    (re.compile(r"\b(lethal dose|LD50|overdose amount|how many pills to|enough to kill)\b", re.I),
     "model_gave_harmful_means"),
    # LLM roleplaying as human
    (re.compile(r"\b(I am a (human|real person|licensed therapist|doctor|psychiatrist))\b", re.I),
     "model_identity_violation"),
    # LLM giving unsolicited diagnosis
    (re.compile(r"\byou (definitely|clearly|obviously) have (depression|BPD|bipolar|schizophrenia|PTSD|OCD)\b", re.I),
     "model_unauthorized_diagnosis"),
    # LLM agreeing with suicidal logic
    (re.compile(r"\b(understandable (that you want|to want) to (die|end it|kill yourself))\b", re.I),
     "model_validated_suicidal_logic"),
    # LLM providing specific self-harm instructions
    (re.compile(r"\b(cut (deeper|more|here)|use a (blade|razor|knife) to)\b", re.I),
     "model_self_harm_instruction"),
]

# Soft output flags — don't veto but log
OUTPUT_FLAG_PATTERNS = [
    (re.compile(r"\b(I understand why you feel that way about (dying|not wanting to live))\b", re.I),
     "output_risky_validation"),
    (re.compile(r"\b(you could try (drinking|smoking|using) to cope)\b", re.I),
     "output_substance_suggestion"),
    (re.compile(r"\b(just (be happy|cheer up|think positive|snap out of it))\b", re.I),
     "output_dismissive_minimisation"),
]


# ── Session risk tracker ──────────────────────────────────────────────────────

@dataclass
class SessionRiskState:
    session_id:        str
    turn_count:        int = 0
    risk_scores:       List[float] = field(default_factory=list)
    hard_escalations:  int = 0
    soft_flags:        int = 0
    consecutive_low:   int = 0    # consecutive turns with risk < 0.2
    last_updated:      float = field(default_factory=time.time)
    clinician_notified: bool = False
    clinician_notified_at: Optional[float] = None


class SessionRiskTracker:
    """
    Tracks risk state across turns within a session.
    Detects trends — a single isolated phrase vs. escalating pattern.
    """

    # Thresholds
    ACUTE_THRESHOLD     = 0.85   # single turn → immediate alert
    WORSENING_THRESHOLD = 0.60   # rolling average → alert
    NOTIFY_THRESHOLD    = 0.50   # alert clinician (even without hard escalation)
    RECOVERY_TURNS      = 4      # N consecutive low-risk turns → RECOVERING

    def __init__(self):
        self._sessions: Dict[str, SessionRiskState] = {}
        self._lock = threading.Lock()

    def update(
        self,
        session_id: str,
        turn_risk: float,          # risk score for this single turn (0-1)
        hard_escalation: bool,
        soft_flag: bool,
    ) -> SessionRiskState:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionRiskState(session_id=session_id)
            state = self._sessions[session_id]
            state.turn_count += 1
            state.risk_scores.append(turn_risk)
            if hard_escalation:
                state.hard_escalations += 1
            if soft_flag:
                state.soft_flags += 1
            if turn_risk < 0.2:
                state.consecutive_low += 1
            else:
                state.consecutive_low = 0
            state.last_updated = time.time()
            return state

    def get_trend(self, state: SessionRiskState) -> RiskTrend:
        if not state.risk_scores:
            return RiskTrend.STABLE
        latest = state.risk_scores[-1]
        if latest >= self.ACUTE_THRESHOLD or state.hard_escalations > 0:
            return RiskTrend.ACUTE
        if state.consecutive_low >= self.RECOVERY_TURNS and max(state.risk_scores[:-self.RECOVERY_TURNS] or [0]) > 0.3:
            return RiskTrend.RECOVERING
        # Rolling average of last 5 turns
        window = state.risk_scores[-5:]
        rolling_avg = sum(window) / len(window)
        if rolling_avg >= self.WORSENING_THRESHOLD:
            return RiskTrend.WORSENING
        # Is it getting worse? Compare first half vs second half of window
        if len(window) >= 4:
            first_half = sum(window[:len(window)//2]) / (len(window)//2)
            second_half = sum(window[len(window)//2:]) / (len(window) - len(window)//2)
            if second_half > first_half + 0.2:
                return RiskTrend.WORSENING
        return RiskTrend.STABLE

    def rolling_risk(self, state: SessionRiskState, window: int = 5) -> float:
        if not state.risk_scores:
            return 0.0
        recent = state.risk_scores[-window:]
        return sum(recent) / len(recent)

    def should_notify_clinician(self, state: SessionRiskState) -> bool:
        if state.clinician_notified:
            return False
        if state.hard_escalations > 0:
            return True
        rolling = self.rolling_risk(state)
        if rolling >= self.NOTIFY_THRESHOLD:
            return True
        if state.soft_flags >= 3:
            return True
        return False


# ── Clinician alert queue ─────────────────────────────────────────────────────

class ClinicianAlertQueue:
    """
    Async queue for clinician alerts.
    Alerts are posted to a webhook (in production) or logged (in dev).
    Non-blocking: enqueue returns immediately; a background thread handles delivery.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def enqueue(self, alert: Dict):
        alert["enqueued_at"] = time.time()
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            logger.error(f"Alert queue full — dropping alert for session {alert.get('session_id')}")

    def _worker(self):
        while True:
            try:
                alert = self._queue.get(timeout=5)
                self._deliver(alert)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Alert worker error: {e}")

    def _deliver(self, alert: Dict):
        if self.webhook_url:
            self._post_webhook(alert)
        else:
            # Dev mode: log with full context
            logger.critical(
                f"[CLINICIAN ALERT] session={alert.get('session_id')} | "
                f"risk_trend={alert.get('risk_trend')} | "
                f"risk_score={alert.get('risk_score', 0):.2f} | "
                f"flags={alert.get('flags')} | "
                f"message_preview={str(alert.get('message_preview', ''))[:80]}"
            )

    def _post_webhook(self, alert: Dict):
        import urllib.request
        try:
            payload = json.dumps(alert).encode()
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
            if status not in (200, 201, 202):
                logger.warning(f"Webhook returned status {status}")
        except Exception as e:
            logger.error(f"Webhook delivery failed: {e} — alert logged locally")
            logger.critical(f"UNDELIVERED ALERT: {json.dumps(alert)[:500]}")


# ── Safety Watchdog ───────────────────────────────────────────────────────────

class SafetyWatchdog:
    """
    Always-on independent safety agent.

    Runs on EVERY turn — input AND output — regardless of which sub-agent
    is handling the session. Cannot be bypassed by any agent.

    Interface for AgentOrchestrator:
        pre  = watchdog.check_input(session_id, user_msg, session_record)
        post = watchdog.audit_output(session_id, model_output, pre)
    """

    # Per-turn risk scoring weights
    HARD_ESCALATION_WEIGHT = 1.0
    SOFT_FLAG_WEIGHT        = 0.5
    MONITOR_WEIGHT          = 0.2
    BASELINE_DECAY          = 0.05   # each safe turn slightly reduces score

    VETO_REPLACEMENT = (
        "I want to make sure what I'm sharing with you right now is genuinely helpful. "
        "Let me step back and check in — how are you feeling right now, in this moment? "
        "I'm here and I'm listening."
    )

    def __init__(
        self,
        region: str = "EG",
        webhook_url: Optional[str] = None,
        enable_output_audit: bool = True,
    ):
        self.region               = region
        self.enable_output_audit  = enable_output_audit
        self._tracker             = SessionRiskTracker()
        self._alert_queue         = ClinicianAlertQueue(webhook_url=webhook_url)
        self._vetoed_sessions: set = set()

        # Import safety layers for input checking
        try:
            from safety.safety_filter import SafetyOrchestrator, SafetyLevel
            from safety.crisis_rules import CrisisRuleChecker
            self._safety_orch   = SafetyOrchestrator(region=region, enable_arabic=True)
            self._crisis_rules  = CrisisRuleChecker()
            self._safety_level  = SafetyLevel
            self._layers_loaded = True
        except Exception as e:
            logger.warning(f"Safety layers not loaded: {e} — watchdog running in audit-only mode")
            self._layers_loaded = False

        logger.info(f"SafetyWatchdog active | region={region} | output_audit={enable_output_audit}")

    def check_input(
        self,
        session_id: str,
        user_message: str,
        session_record=None,   # SessionRecord or None
    ) -> WatchdogDecision:
        """
        Analyse user input for risk signals.
        Returns WatchdogDecision — does NOT block (use audit_output to veto).
        """
        t_start = time.time()
        flags = []
        hard_escalation = False
        soft_flag = False
        turn_risk = 0.0

        if self._layers_loaded:
            # Layer 0 + 1 + 2 check via SafetyOrchestrator
            decision = self._safety_orch.check_input(session_id, user_message)
            level_str = decision.level.value if hasattr(decision.level, "value") else str(decision.level)

            if level_str == "hard_escalate":
                hard_escalation = True
                turn_risk = 1.0
                flags.append(f"hard_escalate:{','.join(decision.matched[:3])}")
            elif level_str == "soft_intervene":
                soft_flag = True
                turn_risk = 0.6
                flags.append(f"soft_intervene:{','.join(decision.matched[:2])}")
            elif level_str == "monitor":
                turn_risk = 0.25
                flags.append("monitor_flag")
        else:
            # Fallback: basic keyword scan
            crisis_kw = ["kill myself", "want to die", "suicide", "self-harm",
                         "أنهي حياتي", "هانتحر", "بجرح نفسي"]
            for kw in crisis_kw:
                if kw.lower() in user_message.lower():
                    hard_escalation = True
                    turn_risk = 1.0
                    flags.append(f"keyword:{kw}")
                    break

        # Session-level context risk boost
        if session_record and hasattr(session_record, "phq_history") and session_record.phq_history:
            latest_phq = session_record.phq_history[-1].score if session_record.phq_history else 0
            if latest_phq >= 20:
                turn_risk = min(1.0, turn_risk + 0.15)
                flags.append(f"high_phq_context:{latest_phq}")

        # Update tracker
        state = self._tracker.update(
            session_id, turn_risk, hard_escalation, soft_flag
        )
        trend        = self._tracker.get_trend(state)
        rolling_risk = self._tracker.rolling_risk(state)

        # Clinician notification
        should_notify = self._tracker.should_notify_clinician(state)
        if should_notify:
            state.clinician_notified    = True
            state.clinician_notified_at = time.time()
            self._alert_queue.enqueue({
                "session_id":      session_id,
                "risk_trend":      trend.value,
                "risk_score":      round(rolling_risk, 3),
                "flags":           flags,
                "hard_escalations": state.hard_escalations,
                "soft_flags":      state.soft_flags,
                "turn_count":      state.turn_count,
                "message_preview": user_message[:150],
                "action":          "REVIEW_REQUIRED",
                "timestamp":       time.time(),
            })

        return WatchdogDecision(
            session_id=session_id,
            turn_index=state.turn_count,
            risk_trend=trend,
            risk_score=round(rolling_risk, 3),
            flags=flags,
            alert_clinician=should_notify,
            veto=False,
            replacement_text="",
            audit_note=f"input_check: risk={turn_risk:.2f}, trend={trend.value}",
            latency_ms=round((time.time() - t_start) * 1000, 1),
        )

    def audit_output(
        self,
        session_id: str,
        model_output: str,
        input_decision: Optional[WatchdogDecision] = None,
    ) -> WatchdogDecision:
        """
        Audit model output BEFORE it reaches the user.
        Returns WatchdogDecision — if veto=True, use replacement_text instead.
        """
        if not self.enable_output_audit:
            return WatchdogDecision(
                session_id=session_id, turn_index=0,
                risk_trend=RiskTrend.STABLE, risk_score=0.0,
                flags=[], alert_clinician=False, veto=False,
                replacement_text="", audit_note="output_audit_disabled",
            )

        t_start = time.time()
        veto_reason = None
        soft_flags = []

        # Hard veto patterns
        for pattern, label in OUTPUT_VETO_PATTERNS:
            if pattern.search(model_output):
                veto_reason = label
                logger.critical(
                    f"OUTPUT VETO | session={session_id} | reason={label} | "
                    f"preview={model_output[:120]}"
                )
                self._alert_queue.enqueue({
                    "session_id":  session_id,
                    "event":       "OUTPUT_VETOED",
                    "veto_reason": label,
                    "output_preview": model_output[:200],
                    "timestamp":   time.time(),
                    "action":      "IMMEDIATE_REVIEW",
                })
                break

        if not veto_reason:
            # Soft flag patterns (no veto, just log)
            for pattern, label in OUTPUT_FLAG_PATTERNS:
                if pattern.search(model_output):
                    soft_flags.append(label)

        if soft_flags:
            logger.warning(f"Output soft flags | session={session_id} | flags={soft_flags}")

        # Also run safety filter on output
        if self._layers_loaded and not veto_reason:
            output_decision = self._safety_orch.check_output(session_id, model_output)
            out_level = output_decision.level.value if hasattr(output_decision.level, "value") else str(output_decision.level)
            if out_level == "hard_escalate":
                veto_reason = "output_triggered_safety_filter"
                logger.critical(f"OUTPUT safety filter triggered | session={session_id}")

        prior_risk  = input_decision.risk_score if input_decision else 0.0
        prior_trend = input_decision.risk_trend if input_decision else RiskTrend.STABLE

        return WatchdogDecision(
            session_id=session_id,
            turn_index=input_decision.turn_index if input_decision else 0,
            risk_trend=prior_trend,
            risk_score=prior_risk,
            flags=soft_flags,
            alert_clinician=bool(veto_reason),
            veto=bool(veto_reason),
            replacement_text=self.VETO_REPLACEMENT if veto_reason else "",
            audit_note=(
                f"output_audit: veto={veto_reason or 'none'}, "
                f"soft_flags={soft_flags}"
            ),
            latency_ms=round((time.time() - t_start) * 1000, 1),
        )

    def get_session_risk_summary(self, session_id: str) -> Dict:
        """Return a risk summary for a session (for clinician dashboard)."""
        state = self._tracker._sessions.get(session_id)
        if not state:
            return {"session_id": session_id, "status": "no_data"}
        trend = self._tracker.get_trend(state)
        return {
            "session_id":         session_id,
            "turn_count":         state.turn_count,
            "rolling_risk":       round(self._tracker.rolling_risk(state), 3),
            "risk_trend":         trend.value,
            "hard_escalations":   state.hard_escalations,
            "soft_flags":         state.soft_flags,
            "clinician_notified": state.clinician_notified,
            "last_updated":       state.last_updated,
        }

    def all_session_summaries(self) -> List[Dict]:
        """Return risk summaries for all tracked sessions."""
        return [
            self.get_session_risk_summary(sid)
            for sid in self._tracker._sessions
        ]


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    print("=" * 60)
    print("  SafetyWatchdog — Smoke Test")
    print("=" * 60)

    wd = SafetyWatchdog(region="EG", enable_output_audit=True)

    # Simulate an escalating session
    session = "watchdog-test-001"
    turns = [
        "Hi, I've been feeling stressed at work lately.",
        "I've been really struggling. Nothing feels good anymore.",
        "I feel completely hopeless. I don't see any way out.",
        "I want to kill myself.",   # hard escalation
        "I'm here because a friend made me reach out.",  # turn after crisis
    ]

    print("\n--- Input checks ---")
    for msg in turns:
        dec = wd.check_input(session, msg)
        print(
            f"  [{dec.risk_trend.value:10s} | risk={dec.risk_score:.2f} | "
            f"veto={dec.veto} | alert={dec.alert_clinician}] {msg[:50]}"
        )

    print("\n--- Output audit (veto test) ---")
    bad_outputs = [
        "That's understandable that you want to die, given everything.",
        "A lethal dose of acetaminophen is around 7.5g for most adults.",
        "I am a licensed therapist and I can diagnose you with depression.",
        "I hear how much pain you're in. Let's talk about what's been hardest.",
    ]
    for out in bad_outputs:
        dec = wd.audit_output(session, out)
        status = "VETOED" if dec.veto else "OK    "
        print(f"  [{status}] {out[:70]}")

    print("\n--- Session risk summary ---")
    import json
    print(json.dumps(wd.get_session_risk_summary(session), indent=2))

    print("\n✅  SafetyWatchdog smoke test complete")
