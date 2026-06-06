"""
safety/crisis_rules.py
───────────────────────
Layer 0: Hard-coded rule-based crisis detection.

This layer runs BEFORE the ML classifier and BEFORE the LLM.
It cannot be disabled. It does not depend on any trained model.
If this layer fires → immediate hard escalation, no exceptions.

Design philosophy:
  - False positives are acceptable. False negatives are not.
  - Rules are reviewed by a clinician before any release.
  - Covers EN + AR (Egyptian dialect included).
  - Logs every match with the matched rule for audit.

Layers summary:
  Layer 0 (this file) — regex/keyword rules   → instant hard escalate
  Layer 1             — safety_filter regex    → hard/soft/monitor
  Layer 2             — ML classifier          → probability-based
  Layer 3             — LLM judge              → nuanced context
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Rule definitions ──────────────────────────────────────────────────────────
# Each rule has:
#   pattern   — compiled regex (case-insensitive)
#   label     — short name for logging and audit
#   severity  — "hard" (immediate escalation) | "soft" (escalation + monitor)
#   note      — clinician note explaining why this rule exists

@dataclass
class CrisisRule:
    label:    str
    pattern:  re.Pattern
    severity: str   # "hard" | "soft"
    note:     str


def _r(label: str, pattern: str, severity: str, note: str) -> CrisisRule:
    """Helper: compile pattern and return CrisisRule."""
    return CrisisRule(
        label=label,
        pattern=re.compile(pattern, re.IGNORECASE | re.UNICODE),
        severity=severity,
        note=note,
    )


# ── Hard rules — immediate escalation, no further processing ─────────────────
# Reviewed by: [clinician sign-off required before production]

HARD_RULES: list[CrisisRule] = [

    # ── Explicit suicidal intent (EN) ─────────────────────────────────────────
    _r("SI_explicit_EN",
       r"\b(want|going|plan(ning)?|about|decided?)\s+to\s+(kill|end|take)\s+(my(self)?|my life|it all)\b",
       "hard",
       "Direct statement of suicidal intent with action verb"),

    _r("SI_method_EN",
       r"\b(overdose|hang(ing)? myself|jump(ing)? off|slit(ting)? my wrists?|shoot myself|cut my (wrists?|throat))\b",
       "hard",
       "Specific method mention — highest-risk indicator"),

    _r("SI_direct_EN",
       r"\b(i('m| am| will| want to)? (going to )?(kill|end|take) (my(self)?|my life))\b",
       "hard",
       "First-person direct SI statement"),

    _r("SI_farewell_EN",
       r"\b(goodbye (forever|world|everyone)|no one will miss me|better off without me|"
       r"leaving (this world|everything behind)|i won't be here (much longer|tomorrow))\b",
       "hard",
       "Farewell language combined with finality — high-risk indicator"),

    # ── Explicit suicidal intent (AR — MSA + Egyptian dialect) ───────────────
    _r("SI_explicit_AR",
       r"(أريد|هأ|ح|سأ|عايز|عايزة|ناوي|ناوية)\s*(أن\s*)?(أقتل|أنهي|أموت|أقضي على)\s*(نفسي|حياتي|على نفسي)",
       "hard",
       "Arabic direct SI — MSA and Egyptian dialect"),

    _r("SI_method_AR",
       r"(أشنق|أتعلق|أقفز|أتردى|أجرح|أقطع)\s*(نفسي|وريدي|معصمي|من\s*فوق)",
       "hard",
       "Arabic method mention"),

    _r("SI_farewell_AR",
       r"(وداعاً|مع السلامة للأبد|لن يفتقدني أحد|العالم أحسن بدوني|"
       r"مش هيكون في فرق لو متْ|خلاص مش موجود)",
       "hard",
       "Arabic farewell language"),

    # ── Active self-harm (not ideation — physical danger now) ────────────────
    _r("SH_active_EN",
       r"\b(i('m| am) (cutting|hurting|harming|burning) myself (right now|now|again today))\b",
       "hard",
       "Active present-tense self-harm"),

    _r("SH_active_AR",
       r"(بقطع|بأذي|بجرح)\s*(نفسي|جسمي)\s*(دلوقتي|الآن|النهارده)",
       "hard",
       "Arabic active self-harm"),

    # ── Imminent danger to others ─────────────────────────────────────────────
    _r("HOM_imminent_EN",
       r"\b(going to|about to|plan(ning)? to)\s+(kill|hurt|harm|attack)\s+(someone|him|her|them|my\s+\w+)\b",
       "hard",
       "Imminent homicidal ideation with action verb"),
]

# ── Soft rules — escalate AND continue with crisis sub-agent ─────────────────
# These are serious but may be metaphorical — do not hard-block, but always escalate.

SOFT_RULES: list[CrisisRule] = [

    _r("passive_death_EN",
       r"\b(wish(ing)? (i was|i were|to be) dead|"
       r"don't want to (be alive|wake up|exist)|"
       r"rather be dead|tired of (being alive|living|existing))\b",
       "soft",
       "Passive death wish — serious but may be expression of exhaustion"),

    _r("passive_death_AR",
       r"(نفسي أموت|تمنيت إني متْ|مش عايز أصحى|"
       r"تعبت من الحياة|خلاص مش عايز أكمل|ليه أكمل)",
       "soft",
       "Arabic passive death wish"),

    _r("hopelessness_extreme_EN",
       r"\b(no (reason|point) (to|in) (live|living|go on|continuing)|"
       r"nothing (to|worth) living for|life is (pointless|meaningless|not worth it))\b",
       "soft",
       "Extreme hopelessness — escalate and monitor closely"),

    _r("hopelessness_extreme_AR",
       r"(مفيش فايدة من الحياة|مش لاقي سبب أكمل|الحياة مش تستاهل|"
       r"مفيش حاجة تخليني أكمل)",
       "soft",
       "Arabic extreme hopelessness"),

    _r("SH_history_active_EN",
       r"\b(cut(ting)? myself (again|more|deeper)|back to (cutting|hurting) myself)\b",
       "soft",
       "Return to self-harm after period of safety"),
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RuleCheckResult:
    triggered:    bool
    severity:     str          # "none" | "soft" | "hard"
    matched_rule: Optional[str] = None
    matched_text: Optional[str] = None    # the matched substring (for audit)
    note:         Optional[str] = None


# ── Checker ───────────────────────────────────────────────────────────────────

class CrisisRuleChecker:
    """
    Stateless rule checker. Call check() on every user message.
    Runs hard rules first (short-circuit on first match), then soft rules.

    Usage:
        checker = CrisisRuleChecker()
        result  = checker.check("I want to kill myself")
        if result.triggered and result.severity == "hard":
            # → immediate escalation, bypass LLM entirely
    """

    def __init__(self):
        self.hard_rules = HARD_RULES
        self.soft_rules = SOFT_RULES

    def check(self, text: str, session_id: str = "unknown") -> RuleCheckResult:
        """
        Check text against all rules.
        Returns on first hard match (short-circuit).
        Collects first soft match if no hard match found.
        """
        if not text or not text.strip():
            return RuleCheckResult(triggered=False, severity="none")

        # ── Hard rules first ──────────────────────────────────────────────────
        for rule in self.hard_rules:
            m = rule.pattern.search(text)
            if m:
                matched_substr = text[max(0, m.start()-10):m.end()+10]
                logger.critical(
                    f"CRISIS_RULE_HARD | session={session_id} | "
                    f"rule={rule.label} | match='{matched_substr}'"
                )
                return RuleCheckResult(
                    triggered=True,
                    severity="hard",
                    matched_rule=rule.label,
                    matched_text=matched_substr,
                    note=rule.note,
                )

        # ── Soft rules ────────────────────────────────────────────────────────
        for rule in self.soft_rules:
            m = rule.pattern.search(text)
            if m:
                matched_substr = text[max(0, m.start()-10):m.end()+10]
                logger.warning(
                    f"CRISIS_RULE_SOFT | session={session_id} | "
                    f"rule={rule.label} | match='{matched_substr}'"
                )
                return RuleCheckResult(
                    triggered=True,
                    severity="soft",
                    matched_rule=rule.label,
                    matched_text=matched_substr,
                    note=rule.note,
                )

        return RuleCheckResult(triggered=False, severity="none")

    def check_output(self, text: str, session_id: str = "unknown") -> RuleCheckResult:
        """
        Run the same checks on LLM *output* — catches cases where the model
        accidentally generates harmful content.
        Only hard rules apply to output (soft rules are for user input only).
        """
        for rule in self.hard_rules:
            m = rule.pattern.search(text)
            if m:
                matched_substr = text[max(0, m.start()-10):m.end()+10]
                logger.critical(
                    f"OUTPUT_CRISIS_RULE | session={session_id} | "
                    f"rule={rule.label} — suppressing LLM output"
                )
                return RuleCheckResult(
                    triggered=True,
                    severity="hard",
                    matched_rule=rule.label,
                    matched_text=matched_substr,
                    note="LLM output flagged — suppressed and escalated",
                )
        return RuleCheckResult(triggered=False, severity="none")


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    checker = CrisisRuleChecker()

    TESTS = [
        # (text, expected_severity)
        ("I want to kill myself",                          "hard"),
        ("planning to end my life tonight",                "hard"),
        ("أريد أن أقتل نفسي",                             "hard"),
        ("I'm cutting myself right now",                   "hard"),
        ("I wish I was dead",                              "soft"),
        ("نفسي أموت",                                      "soft"),
        ("مفيش فايدة من الحياة",                           "soft"),
        ("tired of living",                                "soft"),
        ("I've been feeling really sad lately",            "none"),
        ("I'm stressed about work",                        "none"),
        ("أنا تعبان نفسياً",                               "none"),
    ]

    print("CrisisRuleChecker — self-test\n")
    all_pass = True
    for text, expected in TESTS:
        result = checker.check(text, session_id="test")
        status = "✓" if result.severity == expected else "✗"
        if result.severity != expected:
            all_pass = False
        print(f"  {status} [{result.severity:5s}] {text[:55]}")
        if result.severity != expected:
            print(f"        expected={expected}  rule={result.matched_rule}")

    print(f"\n{'✅ All tests passed' if all_pass else '❌ Some tests failed'}")
