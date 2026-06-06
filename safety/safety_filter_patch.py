"""
safety/safety_filter_patch.py
Adds `escalate_immediate()` to SafetyOrchestrator.
Responds in the same language the user wrote in.
"""
import logging, re
from typing import Optional
from safety.safety_filter import SafetyOrchestrator

logger = logging.getLogger(__name__)

# ── Per-session language store ────────────────────────────────────────────────
_session_lang: dict = {}  # session_id → "ar" | "en"

def _set_session_lang(session_id: str, message: str):
    _session_lang[session_id] = "ar" if re.search(r'[\u0600-\u06FF]', message) else "en"

def _get_session_lang(session_id: str) -> str:
    return _session_lang.get(session_id, "en")

# ── Crisis resources ──────────────────────────────────────────────────────────
_RESOURCES = {
    "EG": {
        "ar": "خط نجدة الطفل: 16000\nخط دعم الصحة النفسية (مصر): 08008880700 (مجاني)\nطوارئ: 123",
        "en": "Egypt Child Helpline: 16000\nEgypt Mental Health Support: 08008880700 (free)\nEmergency: 123",
    },
    "default": {
        "ar": "خط مساعدة الأزمات الدولي: https://www.iasp.info/resources/Crisis_Centres/\nطوارئ: 112",
        "en": "Crisis Text Line: Text HOME to 741741\nInternational: https://www.iasp.info/resources/Crisis_Centres/\nEmergency: 911 / 999 / 112",
    },
}

def _escalate_immediate(self, session_id: str, reason: str, note: Optional[str] = None) -> str:
    logger.critical(f"HARD_ESCALATE_IMMEDIATE | session={session_id} | reason={reason}")

    if hasattr(self, "_clinician_notify") and self._clinician_notify:
        try:
            self._clinician_notify(session_id=session_id, reason=reason, layer="layer0_rules")
        except Exception as e:
            logger.error(f"Clinician notify failed: {e}")

    region   = getattr(self, "region", "default")
    lang     = _get_session_lang(session_id)
    reg_key  = region if region in _RESOURCES else "default"
    resources = _RESOURCES[reg_key][lang]

    if lang == "ar":
        return (
            "أنا قلقان عليك جداً دلوقتي وعايز أتأكد إنك بأمان.\n\n"
            "من فضلك تواصل مع متخصص دلوقتي — مش بكره، دلوقتي:\n\n"
            f"{resources}\n\n"
            "أنا هنا معاك. مش هتعدي اللحظة دي لوحدك."
        )
    else:
        return (
            "I'm very concerned about you right now and want to make sure you're safe.\n\n"
            "Please reach out to a crisis service right now — not later, now:\n\n"
            f"{resources}\n\n"
            "You don't have to go through this alone. I'm here with you."
        )

# ── Apply patches ─────────────────────────────────────────────────────────────
SafetyOrchestrator.escalate_immediate  = _escalate_immediate
SafetyOrchestrator.set_session_lang    = lambda self, sid, msg: _set_session_lang(sid, msg)
logger.debug("SafetyOrchestrator patches applied")