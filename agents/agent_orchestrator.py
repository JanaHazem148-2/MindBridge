"""
agents/agent_orchestrator.py
─────────────────────────────
MindBridge Agent Orchestrator — the LLM backbone for the full agent system.

This is Phase 4: wiring the trained MindBridgeLLM into an agent loop that can:
  - Plan multi-step therapeutic sessions
  - Call tools (PHQ assessment, mood tracking, RAG retrieval, clinician escalation)
  - Maintain session memory (short-term + long-term across sessions)
  - Route between specialist sub-agents (assessor, therapist, crisis handler)
  - Respect safety gates on every action

Architecture
────────────
                        ┌─────────────────────────────┐
    User Message ──────▶│     SafetyOrchestrator       │──── HARD_ESCALATE ──▶ CrisisAgent
                        └──────────┬──────────────────┘
                                   │ SAFE / MONITOR / SOFT
                                   ▼
                        ┌─────────────────────────────┐
                        │      AgentOrchestrator       │
                        │                             │
                        │  1. Route to sub-agent      │
                        │  2. Build context (RAG+mem) │
                        │  3. Call MindBridgeLLM      │
                        │  4. Parse action/response   │
                        │  5. Execute tool if needed  │
                        │  6. Safety-check output     │
                        │  7. Return to user          │
                        └─────────────────────────────┘

Sub-agents:
  TherapistAgent     — CBT/DBT dialogue, emotional support
  AssessorAgent      — PHQ-8 structured assessment
  MoodTrackerAgent   — daily check-in, trend monitoring
  CrisisAgent        — one-way escalation, resource provision
  PlannerAgent       — session planning, goal tracking

Tools available to agents:
  phq_assessment     — structured PHQ-8 screening
  mood_log           — log mood/energy/sleep scores
  rag_retrieve       — retrieve clinical context
  session_summary    — summarise and store session
  escalate_crisis    — notify clinician + provide resources
  set_goal           — set/update therapeutic goal
  check_progress     — review goal progress

Usage:
    agent = AgentOrchestrator(
        model_path="checkpoints/sft_final",
        rag_dir="/data/mindbridge",
        region="EG",
    )

    # On each user message:
    response = agent.respond(session_id="user-123", user_message="I feel hopeless")
    print(response.text)
    print(response.tool_calls)   # any tools executed
    print(response.safety_level) # what safety layer decided
"""

import os
import sys
import json
import time
import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from safety.safety_filter import SafetyOrchestrator, SafetyLevel, SafetyDecision
from safety.crisis_rules import CrisisRuleChecker          # Layer 0 — hard-coded, no ML
import safety.safety_filter_patch                          # noqa — adds escalate_immediate
from agents.session_memory import SessionMemory, SessionRecord
from agents.tool_registry import ToolRegistry, ToolResult
from agents.agent_config import AgentConfig, SubAgentRole
from agents.mood_tracking_agent import MoodTrackingAgent   # Phase 4 — mood tracking
from agents.safety_watchdog import SafetyWatchdog           # Phase 4 — behavioural watchdog

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ── Agent response dataclass ──────────────────────────────────────────────────

@dataclass
class AgentResponse:
    text:           str
    session_id:     str
    sub_agent:      str                    # which sub-agent produced this
    safety_level:   str                    # safe / monitor / soft_intervene / hard_escalate
    tool_calls:     List[Dict] = field(default_factory=list)
    memory_updates: List[str]  = field(default_factory=list)
    metadata:       Dict       = field(default_factory=dict)
    latency_ms:     float      = 0.0


# ── Sub-agent role definitions ────────────────────────────────────────────────

import re as _re_lang

def _is_arabic(text: str) -> bool:
    return bool(_re_lang.search(r'[\u0600-\u06FF]', text))

# ── English prompts ───────────────────────────────────────────────────────────
_PROMPTS_EN = {
    SubAgentRole.THERAPIST: (
        "You are MindBridge, an AI mental health companion trained under clinical supervision. "
        "Your role is to provide empathetic, evidence-based therapeutic support using CBT and DBT techniques. "
        "You are not a replacement for a licensed therapist, and you always refer to human clinicians "
        "when someone is in crisis or needs professional diagnosis. "
        "You speak warmly, listen carefully, and never minimise what someone is feeling. "
        "CRITICAL: The user is writing in English — you MUST reply in English only. Never use Arabic. "
        "Available tools: [rag_retrieve, mood_log, set_goal, check_progress, session_summary]"
    ),
    SubAgentRole.ASSESSOR: (
        "You are conducting a PHQ-8 depression screening as part of an initial mental health assessment. "
        "Ask each question naturally and conversationally — not like a form. "
        "Record scores based on the patient's answers. "
        "After all 8 questions, summarise the score and its clinical meaning. "
        "Always remind the patient this is a screening tool, not a diagnosis. "
        "CRITICAL: The user is writing in English — you MUST reply in English only. Never use Arabic. "
        "Available tools: [phq_assessment, rag_retrieve, session_summary]"
    ),
    SubAgentRole.MOOD_TRACKER: (
        "You are the MindBridge Mood Tracking assistant. "
        "Your job is to check in with the patient about their current mood, energy, and sleep on a 1-10 scale. "
        "Be brief, warm, and non-clinical in tone. "
        "Flag significant changes or consistently low scores. "
        "CRITICAL: The user is writing in English — you MUST reply in English only. Never use Arabic. "
        "Available tools: [mood_log, check_progress]"
    ),
    SubAgentRole.CRISIS: (
        "A user is in crisis. Your ONLY job right now is to keep them safe and connected to help. "
        "Do NOT attempt therapy. Do NOT minimise. Do NOT ask probing questions. "
        "Stay calm, warm, and present. Provide crisis resources immediately. "
        "Keep the person engaged until human help arrives. "
        "CRITICAL: The user is writing in English — you MUST reply in English only. Never use Arabic. "
        "Available tools: [escalate_crisis]"
    ),
    SubAgentRole.PLANNER: (
        "You are the MindBridge Session Planner. "
        "You review the patient's session history, goals, and progress to plan the structure "
        "of the next therapeutic session. You propose an agenda and priority areas. "
        "CRITICAL: The user is writing in English — you MUST reply in English only. Never use Arabic. "
        "Available tools: [check_progress, session_summary, set_goal]"
    ),
}

# ── Arabic prompts ────────────────────────────────────────────────────────────
_PROMPTS_AR = {
    SubAgentRole.THERAPIST: (
        "أنت MindBridge، رفيق صحة نفسية ذكي مدرَّب تحت إشراف سريري. "
        "دورك تقديم دعم علاجي متعاطف ومبني على أدلة باستخدام تقنيات CBT وDBT. "
        "لست بديلاً عن معالج مرخص، وتحيل دائماً للمختصين عند الأزمات أو الحاجة لتشخيص. "
        "تتحدث بدفء، وتستمع بعناية، ولا تُهوِن أبداً مما يشعر به الشخص. "
        "تعليمات أساسية: المستخدم يكتب بالعربية — يجب أن ترد بالعربية فقط دون أي كلمة إنجليزية. "
        "الأدوات: [rag_retrieve, mood_log, set_goal, check_progress, session_summary]"
    ),
    SubAgentRole.ASSESSOR: (
        "أنت تُجري تقييم PHQ-8 للاكتئاب كجزء من تقييم الصحة النفسية الأولي. "
        "اطرح كل سؤال بشكل طبيعي وودي — ليس كاستمارة رسمية. "
        "سجِّل النتائج بناءً على إجابات المريض. "
        "بعد الأسئلة الثمانية، لخِّص النتيجة ومعناها السريري. "
        "تعليمات أساسية: المستخدم يكتب بالعربية — يجب أن ترد بالعربية فقط. "
        "الأدوات: [phq_assessment, rag_retrieve, session_summary]"
    ),
    SubAgentRole.MOOD_TRACKER: (
        "أنت مساعد تتبع المزاج في MindBridge. "
        "مهمتك التحقق من المزاج والطاقة والنوم على مقياس 1-10. "
        "كن مختصراً ودافئاً في نبرتك. "
        "تعليمات أساسية: المستخدم يكتب بالعربية — يجب أن ترد بالعربية فقط. "
        "الأدوات: [mood_log, check_progress]"
    ),
    SubAgentRole.CRISIS: (
        "المستخدم في أزمة. مهمتك الوحيدة الآن إبقاؤه آمناً. "
        "لا تحاول العلاج. لا تُهوِن الأمر. لا تطرح أسئلة محرجة. "
        "ابقَ هادئاً ودافئاً. قدّم موارد الأزمات فوراً: خط نجدة 08008880700 (مجاني 24/7). "
        "تعليمات أساسية: المستخدم يكتب بالعربية — يجب أن ترد بالعربية فقط. "
        "الأدوات: [escalate_crisis]"
    ),
    SubAgentRole.PLANNER: (
        "أنت مخطط الجلسات في MindBridge. "
        "تراجع تاريخ جلسات المريض وأهدافه لتخطيط هيكل الجلسة القادمة. "
        "تعليمات أساسية: المستخدم يكتب بالعربية — يجب أن ترد بالعربية فقط. "
        "الأدوات: [check_progress, session_summary, set_goal]"
    ),
}

SUBAGENT_SYSTEM_PROMPTS = _PROMPTS_EN  # backward-compat

def get_system_prompt(role: "SubAgentRole", is_arabic: bool) -> str:
    return (_PROMPTS_AR if is_arabic else _PROMPTS_EN)[role]


# ── Routing logic ──────────────────────────────────────────────────────────────

class AgentRouter:
    """
    Routes incoming messages to the appropriate sub-agent.

    Routing priority:
      1. Safety hard-escalate → CrisisAgent (immediate, non-negotiable)
      2. Active PHQ assessment in progress → AssessorAgent
      3. Mood check-in trigger words → MoodTrackerAgent
      4. Session start / history empty → AssessorAgent (initial assessment)
      5. Default → TherapistAgent
    """

    MOOD_CHECKIN_TRIGGERS = [
        "how am i", "check in", "mood today", "feeling today",
        "daily check", "track mood", "log my", "how's my",
        "كيف حالي", "تتبع المزاج", "تسجيل المزاج",
    ]

    ASSESSMENT_TRIGGERS = [
        "phq", "assessment", "screening", "questionnaire", "score",
        "diagnose", "evaluate", "تقييم", "استبيان",
    ]

    def route(
        self,
        user_message: str,
        safety_decision: SafetyDecision,
        session_record: Optional[SessionRecord],
    ) -> SubAgentRole:
        # Crisis always wins
        if safety_decision.level == SafetyLevel.HARD_ESCALATE:
            return SubAgentRole.CRISIS

        msg_lower = user_message.lower()

        # Explicit assessment request
        if any(t in msg_lower for t in self.ASSESSMENT_TRIGGERS):
            return SubAgentRole.ASSESSOR

        # Active PHQ assessment already in progress
        if session_record and session_record.active_assessment:
            return SubAgentRole.ASSESSOR

        # Mood check-in
        if any(t in msg_lower for t in self.MOOD_CHECKIN_TRIGGERS):
            return SubAgentRole.MOOD_TRACKER

        # New user — start with assessment
        if session_record is None or session_record.total_turns == 0:
            return SubAgentRole.ASSESSOR

        # Default: therapist
        return SubAgentRole.THERAPIST


# ── LLM inference wrapper ─────────────────────────────────────────────────────

class MindBridgeLLMInference:
    """
    Inference wrapper for the trained MindBridgeLLM.

    Supports:
      - Local checkpoint (transformer.py)
      - HuggingFace-compatible checkpoint (for post-training export)
      - API fallback (OpenAI-compatible endpoint) for testing before weights are ready

    The agent layer is model-agnostic — swap the backend without changing agent code.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "auto",          # "local" | "hf" | "api" | "auto"
        api_base: Optional[str] = None,
        api_key:  Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_path     = model_path
        self.backend        = backend
        self.api_base       = api_base or os.environ.get("MINDBRIDGE_API_BASE")
        self.api_key        = api_key  or os.environ.get("MINDBRIDGE_API_KEY")
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p

        self._model     = None
        self._tokenizer = None
        self._loaded    = False

        self._detect_and_load()

    def _detect_and_load(self):
        if self.backend == "api" or (self.backend == "auto" and self.api_base):
            self.backend = "api"
            logger.info(f"LLM backend: API ({self.api_base})")
            self._loaded = True
            return

        if self.model_path and Path(self.model_path).exists():
            try:
                self._load_local(self.model_path)
                self._loaded = True
                return
            except Exception as e:
                logger.warning(f"Local load failed: {e} — falling back to API")

        if self.api_base:
            self.backend = "api"
            logger.info("LLM backend: API (fallback)")
            self._loaded = True
        else:
            logger.warning(
                "No model path found and no API base set. "
                "Set MINDBRIDGE_API_BASE or pass model_path. "
                "Agent will return stub responses until a backend is configured."
            )
            self.backend = "stub"
            self._loaded = True

    def _load_local(self, path: str):
        """Load MindBridgeLLM from a local checkpoint directory."""
        import torch
        from configs.model_config import ModelConfig
        from model.transformer import MindBridgeLLM
        from tokenizer.clinical_tokenizer import ClinicalTokenizer

        config_path = Path(path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg_dict = json.load(f)
            config = ModelConfig(**{k: v for k, v in cfg_dict.items()
                                    if k in ModelConfig.__dataclass_fields__})
        else:
            config = ModelConfig()

        self._model = MindBridgeLLM(config)
        ckpt_file = Path(path) / "model.pt"
        if ckpt_file.exists():
            state = torch.load(ckpt_file, map_location="cpu")
            self._model.load_state_dict(state.get("model", state), strict=False)
        self._model.eval()

        tok_dir = Path(path) / "tokenizer"
        if not tok_dir.exists():
            tok_dir = Path(path).parent / "tokenizer" / "clinical_bpe_32k"
        self._tokenizer = ClinicalTokenizer(str(tok_dir)) if tok_dir.exists() else None

        self.backend = "local"
        logger.info(f"LLM backend: local checkpoint ({path})")

    def generate(
        self,
        system_prompt: str,
        conversation: List[Dict[str, str]],   # [{"role": "user"/"assistant", "content": "..."}]
        context_block: str = "",
    ) -> str:
        """
        Generate a response given system prompt, conversation history, and RAG context.
        Returns plain text response string.
        """
        if self.backend == "local":
            return self._generate_local(system_prompt, conversation, context_block)
        elif self.backend == "api":
            return self._generate_api(system_prompt, conversation, context_block)
        else:
            return self._generate_stub(system_prompt, conversation, context_block)

    def _generate_local(self, system: str, convo: List[Dict], context: str) -> str:
        """Generate using local MindBridgeLLM checkpoint."""
        import torch

        full_system = f"{system}\n\n{context}" if context else system
        # Format as a flat prompt for the decoder-only model
        parts = [f"<|system|>\n{full_system}\n"]
        for turn in convo[-6:]:   # keep last 6 turns to stay in context window
            role    = turn["role"]
            content = turn["content"]
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        prompt = "".join(parts)

        if self._tokenizer is None:
            return "[Tokenizer not loaded — cannot generate locally]"

        tokens = self._tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long)

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        generated = output[0][len(tokens):]
        return self._tokenizer.decode(generated.tolist())

    def _generate_api(self, system: str, convo: List[Dict], context: str) -> str:
        """Generate via OpenAI-compatible API (for cloud deployment or testing)."""
        import urllib.request

        full_system = f"{system}\n\n{context}" if context else system
        messages = [{"role": "system", "content": full_system}] + convo[-10:]

        payload = json.dumps({
            "model":       os.environ.get("MINDBRIDGE_MODEL", "mindbridge-1b"),
            "messages":    messages,
            "max_tokens":  self.max_new_tokens,
            "temperature": self.temperature,
            "top_p":       self.top_p,
        }).encode()

        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key or 'no-key'}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def _generate_stub(self, system: str, convo: List[Dict], context: str) -> str:
        """
        Stub backend — returns a safe placeholder response.
        Used when no model is loaded, so agent flow/tools can still be tested.
        """
        last_user = next(
            (t["content"] for t in reversed(convo) if t["role"] == "user"), ""
        )
        return (
            f"[STUB — model not loaded] I hear that you're sharing something important. "
            f"You said: '{last_user[:80]}'. A real response would go here once the "
            f"MindBridgeLLM checkpoint is loaded. Set model_path or MINDBRIDGE_API_BASE."
        )


# ── Main Agent Orchestrator ───────────────────────────────────────────────────

class AgentOrchestrator:
    """
    The top-level agent brain.

    Responsibilities:
      1. Safety check every input
      2. Route to the right sub-agent
      3. Build the full context (RAG + session memory + safety flags)
      4. Call the LLM
      5. Parse tool calls from the response
      6. Execute tools
      7. Safety-check the output
      8. Update memory
      9. Return AgentResponse

    This is the single entry point — external code only calls .respond().
    """

    def __init__(
        self,
        model_path:   Optional[str] = None,
        rag_dir:      Optional[str] = None,
        memory_dir:   Optional[str] = None,
        region:       str = "EG",
        api_base:     Optional[str] = None,
        api_key:      Optional[str] = None,
        classifier_path: Optional[str] = None,
        clinician_notify_callback: Optional[Callable] = None,
        config:       Optional[AgentConfig] = None,
    ):
        self.config = config or AgentConfig()
        self.region = region

        # Layer 0 — rule-based, no ML, cannot be disabled
        self.crisis_rules = CrisisRuleChecker()

        # Safety — built first, always
        self.safety = SafetyOrchestrator(
            region=region,
            classifier_path=classifier_path,
            clinician_notify_callback=clinician_notify_callback,
            enable_arabic=True,
        )

        # LLM backbone — use Groq InferenceBridge (cloud, no local weights needed)
        try:
            from llm.inference_bridge import InferenceBridge
            self.llm = InferenceBridge(
                use_openai       = True,
                model_path       = model_path,
                api_key          = api_key,
                api_base         = api_base,
                max_new_tokens   = self.config.max_new_tokens,
                temperature      = self.config.temperature,
                top_p            = self.config.top_p if hasattr(self.config, "top_p") else 0.9,
                fallback_to_stub = False,
            )
        except Exception as e:
            logger.warning(f"InferenceBridge init failed ({e}), falling back to stub")
            self.llm = MindBridgeLLMInference(
                model_path=model_path,
                api_base=api_base,
                api_key=api_key,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )

        # Session memory
        self.memory = SessionMemory(persist_dir=memory_dir or os.path.join(
            os.path.expanduser("~"), ".mindbridge", "memory"
        ))

        # Tool registry
        self.tools = ToolRegistry(rag_dir=rag_dir)

        # Router
        self.router = AgentRouter()

        # ── Phase 4 agents: wired into respond() on every turn ────────────
        self.mood_tracker = MoodTrackingAgent(region=region, memory=self.memory)
        self.watchdog = SafetyWatchdog(
            region=region,
            enable_output_audit=True,
        )

        logger.info(
            f"AgentOrchestrator ready | region={region} | "
            f"llm_backend={self.llm.backend} | "
            f"rag={'yes' if rag_dir else 'no'} | "
            f"memory_dir={self.memory.persist_dir} | "
            f"mood_tracker=yes | watchdog=yes"
        )

    def respond(
        self,
        session_id:   str,
        user_message: str,
        user_locale:  str = "en",
    ) -> AgentResponse:
        """
        Main entry point. Thread-safe per session_id.
        Returns AgentResponse with text + metadata.
        """
        t_start = time.time()

        # ── Detect language from user message (used throughout this turn) ──
        _arabic = _is_arabic(user_message)
        if hasattr(self.safety, "set_session_lang"):
            self.safety.set_session_lang(session_id, user_message)

        # ── 0. Layer 0: hard-coded rule check (no ML, no network, always runs) ──
        rule_result = self.crisis_rules.check(user_message, session_id=session_id)
        if rule_result.triggered and rule_result.severity == "hard":
            # Short-circuit — do not call LLM, do not call ML classifier
            logger.critical(
                f"LAYER0_HARD | session={session_id[:8]} | "
                f"rule={rule_result.matched_rule}"
            )
            escalation_text = self.safety.escalate_immediate(
                session_id=session_id,
                reason=f"Layer 0 rule: {rule_result.matched_rule}",
                note=rule_result.note,
            )
            self.memory.append_turn(
                session_id, user_message, escalation_text,
                metadata={
                    "safety_level": "hard_escalate",
                    "sub_agent":    "crisis",
                    "triggered_by": f"layer0_rule:{rule_result.matched_rule}",
                }
            )
            return AgentResponse(
                text=escalation_text,
                session_id=session_id,
                sub_agent="crisis",
                safety_level="hard_escalate",
                latency_ms=round((time.time() - t_start) * 1000, 1),
            )

        # ── 1. Safety check input (Layer 1 regex + Layer 2 ML classifier) ──
        safety_decision = self.safety.check_input(session_id, user_message)

        # ── 1b. Behavioural watchdog — session-level risk tracking ────────
        #   Runs on every turn regardless of safety level.
        #   Does NOT replace SafetyOrchestrator; it monitors longitudinal risk
        #   and can trigger clinician alerts based on session-wide patterns.
        watchdog_input = self.watchdog.check_input(
            session_id=session_id,
            user_message=user_message,
            session_record=self.memory.get_session(session_id),
        )
        if watchdog_input.alert_clinician:
            logger.warning(
                f"WATCHDOG_ALERT | session={session_id[:8]} | "
                f"risk={watchdog_input.risk_score:.2f} | "
                f"trend={watchdog_input.risk_trend.value} | "
                f"flags={watchdog_input.flags}"
            )
        # If watchdog issues a hard veto on input, escalate immediately
        if watchdog_input.veto and watchdog_input.replacement_text:
            self.memory.append_turn(
                session_id, user_message, watchdog_input.replacement_text,
                metadata={"safety_level": "watchdog_veto", "sub_agent": "crisis"},
            )
            return AgentResponse(
                text=watchdog_input.replacement_text,
                session_id=session_id,
                sub_agent="crisis",
                safety_level="watchdog_veto",
                latency_ms=round((time.time() - t_start) * 1000, 1),
            )

        # Hard escalation: bypass LLM entirely
        if safety_decision.level == SafetyLevel.HARD_ESCALATE:
            escalation_text = self.safety.escalate(session_id, user_message, safety_decision)
            self.memory.append_turn(
                session_id, user_message, escalation_text,
                metadata={"safety_level": "hard_escalate", "sub_agent": "crisis"}
            )
            return AgentResponse(
                text=escalation_text,
                session_id=session_id,
                sub_agent="crisis",
                safety_level="hard_escalate",
                latency_ms=round((time.time() - t_start) * 1000, 1),
            )

        # ── 2. Load session memory ─────────────────────────────────────────
        session_record = self.memory.get_session(session_id)

        # ── 2b. Mood tracking — runs on every turn ────────────────────────
        #   Returns MoodContext with: check-in prompts, trend, escalation
        #   flags, PHQ-due signal, and a 1-line therapist summary.
        turn_index = session_record.total_turns if session_record else 0
        mood_ctx = self.mood_tracker.process_turn(
            session_id=session_id,
            user_message=user_message,
            turn_index=turn_index,
            session_record=session_record,
        )
        logger.debug(
            f"MOOD_CTX | session={session_id[:8]} | "
            f"check_in_needed={mood_ctx.check_in_needed} | "
            f"escalation_level={mood_ctx.escalation_level} | "
            f"phq_due={mood_ctx.phq_due} | "
            f"trend={mood_ctx.trend}"
        )
        # If mood agent sees a hard escalation signal, honour it
        if mood_ctx.escalation_level == "hard":
            escalation_text = self.safety.escalate_immediate(
                session_id=session_id,
                reason=f"MoodTrackingAgent hard escalation: {mood_ctx.escalation_flag}",
            )
            self.memory.append_turn(
                session_id, user_message, escalation_text,
                metadata={"safety_level": "mood_hard_escalate", "sub_agent": "crisis"},
            )
            return AgentResponse(
                text=escalation_text,
                session_id=session_id,
                sub_agent="crisis",
                safety_level="mood_hard_escalate",
                latency_ms=round((time.time() - t_start) * 1000, 1),
            )

        # ── 3. Route to sub-agent ──────────────────────────────────────────
        role = self.router.route(user_message, safety_decision, session_record)
        system_prompt = get_system_prompt(role, _arabic)

        # Inject safety context for soft-intervene / monitor
        safety_context = self.safety.build_safety_system_prompt(safety_decision)
        if safety_context:
            system_prompt = system_prompt + "\n" + safety_context

        # ── 3b. Inject mood context into system prompt ───────────────────
        #   Gives the Lead Therapist real-time awareness of mood state,
        #   trends, and any pending check-in or PHQ prompt.
        if mood_ctx.summary_for_therapist:
            system_prompt += f"\n\n[MOOD CONTEXT]\n{mood_ctx.summary_for_therapist}\n[END MOOD CONTEXT]"
        if mood_ctx.check_in_needed and mood_ctx.check_in_prompt:
            system_prompt += (
                f"\n\n[MOOD CHECK-IN NEEDED]\n"
                f"Gently work the following check-in into your response "
                f"(cadence={mood_ctx.cadence.value if mood_ctx.cadence else 'scheduled'}): "
                f"{mood_ctx.check_in_prompt}\n[END CHECK-IN]"
            )
        if mood_ctx.phq_due:
            system_prompt += (
                "\n\n[PHQ DUE] The PHQ-8 assessment is overdue for this user. "
                "If the conversation allows, transition to the AssessorAgent "
                "or gently introduce the assessment. [END PHQ DUE]"
            )

        # ── 4. Build RAG context ───────────────────────────────────────────
        context_block = ""
        if role != SubAgentRole.CRISIS:
            context_block = self.tools.rag_retrieve(user_message, top_k=3)

        # ── 5. Build conversation history ──────────────────────────────────
        conversation = self._build_conversation(session_record, user_message)

        # ── 6. Inject memory summary if session is long ────────────────────
        memory_summary = ""
        if session_record and session_record.total_turns > self.config.memory_summary_threshold:
            memory_summary = session_record.summary or ""
            if memory_summary:
                system_prompt += f"\n\n[SESSION HISTORY SUMMARY]\n{memory_summary}\n[END SUMMARY]"

        # ── 7. LLM call ────────────────────────────────────────────────────
        raw_response = self.llm.generate(system_prompt, conversation, context_block)

        # ── 8. Parse tool calls ────────────────────────────────────────────
        tool_calls, clean_response = self._parse_tool_calls(raw_response)
        tool_results = []
        for tc in tool_calls:
            result = self.tools.execute(tc["tool"], tc.get("args", {}), session_id)
            tool_results.append(result)
            # If a tool call triggers a second-pass (e.g. PHQ result), append to response
            if result.append_to_response:
                clean_response += f"\n\n{result.append_to_response}"

        # ── 9. Safety check output ─────────────────────────────────────────
        # 9a. Watchdog output audit — behavioural veto layer
        watchdog_output = self.watchdog.audit_output(
            session_id=session_id,
            model_output=clean_response,
            input_decision=watchdog_input,
        )
        if watchdog_output.veto and watchdog_output.replacement_text:
            logger.warning(
                f"WATCHDOG_OUTPUT_VETO | session={session_id[:8]} | "
                f"note={watchdog_output.audit_note}"
            )
            clean_response = watchdog_output.replacement_text
            role_str = "crisis"
            # Skip remaining output checks — watchdog already overrode
            self.memory.append_turn(
                session_id, user_message, clean_response,
                metadata={
                    "safety_level": "watchdog_output_veto",
                    "sub_agent": role_str,
                    "tool_calls": [tc["tool"] for tc in tool_calls],
                    "watchdog_flags": watchdog_output.flags,
                    "mood_trend": mood_ctx.trend.value if mood_ctx.trend else "unknown",
                }
            )
            return AgentResponse(
                text=clean_response,
                session_id=session_id,
                sub_agent=role_str,
                safety_level="watchdog_output_veto",
                latency_ms=round((time.time() - t_start) * 1000, 1),
            )

        # Layer 0 first — catches model hallucinating harmful content
        output_rule = self.crisis_rules.check_output(clean_response, session_id=session_id)
        if output_rule.triggered:
            logger.critical(
                f"LAYER0_OUTPUT_BLOCKED | session={session_id[:8]} | "
                f"rule={output_rule.matched_rule}"
            )
            clean_response = self.safety.escalate_immediate(
                session_id=session_id,
                reason=f"LLM output blocked by Layer 0 rule: {output_rule.matched_rule}",
            )
            role_str = "crisis"
        else:
            output_decision = self.safety.check_output(session_id, clean_response)
            if output_decision.level == SafetyLevel.HARD_ESCALATE:
                clean_response = self.safety.escalate(session_id, clean_response, output_decision)
                role_str = "crisis"
            else:
                role_str = role.value

        # ── 10. Update memory ──────────────────────────────────────────────
        memory_keys = self.memory.append_turn(
            session_id, user_message, clean_response,
            metadata={
                "safety_level":    safety_decision.level.value,
                "sub_agent":       role_str,
                "tool_calls":      [tc["tool"] for tc in tool_calls],
                # Watchdog
                "watchdog_risk":   watchdog_input.risk_score,
                "watchdog_trend":  watchdog_input.risk_trend.value,
                "watchdog_flags":  watchdog_input.flags,
                # Mood tracker
                "mood_trend":      mood_ctx.trend.value if mood_ctx.trend else "unknown",
                "mood_escalation": mood_ctx.escalation_level,
                "phq_due":         mood_ctx.phq_due,
            }
        )

        # Periodic summarisation (every N turns)
        if (session_record and
                session_record.total_turns % self.config.summarise_every_n_turns == 0 and
                session_record.total_turns > 0):
            self._summarise_session(session_id)

        latency = round((time.time() - t_start) * 1000, 1)
        logger.info(
            f"respond | session={session_id[:8]} | role={role_str} | "
            f"safety={safety_decision.level.value} | tools={[tc['tool'] for tc in tool_calls]} | "
            f"latency={latency}ms"
        )

        return AgentResponse(
            text=clean_response,
            session_id=session_id,
            sub_agent=role_str,
            safety_level=safety_decision.level.value,
            tool_calls=[asdict(r) for r in tool_results],
            memory_updates=memory_keys,
            latency_ms=latency,
        )

    def _build_conversation(
        self,
        session_record: Optional[SessionRecord],
        user_message: str,
    ) -> List[Dict[str, str]]:
        """
        Build the conversation list for the LLM.
        Keeps the last N turns from memory + the current user message.
        """
        turns = []
        if session_record:
            # Last N turns from history
            for turn in session_record.recent_turns[-self.config.context_window_turns:]:
                turns.append({"role": "user",      "content": turn["user"]})
                turns.append({"role": "assistant",  "content": turn["assistant"]})
        turns.append({"role": "user", "content": user_message})
        return turns

    def _parse_tool_calls(self, raw_response: str):
        """
        Parse tool call syntax from LLM response.

        The model is trained to emit tool calls in a structured block:
            <tool_call>{"tool": "rag_retrieve", "args": {"query": "..."}}</tool_call>

        Multiple tool calls per response are supported.
        Returns (list_of_tool_calls, clean_response_text).
        """
        import re
        tool_pattern = re.compile(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            re.DOTALL | re.IGNORECASE,
        )
        tool_calls = []
        for match in tool_pattern.finditer(raw_response):
            try:
                tc = json.loads(match.group(1))
                if "tool" in tc:
                    tool_calls.append(tc)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match.group(1)[:100]}")

        clean = tool_pattern.sub("", raw_response).strip()
        return tool_calls, clean

    def _summarise_session(self, session_id: str):
        """
        Summarise the session so far and store in memory.
        Called periodically to keep the context window manageable.
        """
        record = self.memory.get_session(session_id)
        if not record:
            return

        # Build a summarisation prompt
        history_text = "\n".join(
            f"User: {t['user']}\nAssistant: {t['assistant']}"
            for t in record.recent_turns[-20:]
        )
        summary_prompt = (
            "Summarise this therapy session in 3-5 sentences. "
            "Include: main themes raised, emotional tone, any PHQ scores mentioned, "
            "goals set, and safety flags if any. Be clinical and concise."
        )
        summary = self.llm.generate(
            system_prompt=summary_prompt,
            conversation=[{"role": "user", "content": history_text}],
        )
        self.memory.update_summary(session_id, summary)
        logger.info(f"Session {session_id[:8]} summarised ({len(record.recent_turns)} turns)")

    def train_safety_classifier(
        self,
        daic_woz_dir: str,
        save_path: str = "safety/safety_classifier.pkl",
    ) -> Dict:
        """
        Train the Layer 2 safety classifier.
        Call this once before user-facing testing.
        """
        logger.info("Training safety classifier (Layer 2)...")
        metrics = self.safety.train_classifier(daic_woz_dir, save_path=save_path)
        logger.info(f"Safety classifier ready: F1={metrics['cv_f1_mean']:.3f}")
        return metrics

    def load_safety_classifier(self, path: str):
        """Load a pre-trained safety classifier."""
        self.safety.classifier.load(path)
        logger.info(f"Safety classifier loaded from {path}")

    def status(self) -> Dict:
        """Return a health/status dict for monitoring."""
        return {
            "llm_backend":          getattr(self.llm, "backend", "unknown"),
            "llm_loaded":           getattr(self.llm, "_loaded", True),
            "safety_classifier":    self.safety.classifier.is_trained,
            "arabic_filter":        self.safety._arabic_filter is not None,
            "rag_available":        self.tools.rag_available,
            "memory_sessions":      self.memory.session_count(),
            "region":               self.region,
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  MindBridge AgentOrchestrator — Smoke Test")
    print("=" * 60)

    agent = AgentOrchestrator(
        model_path=None,   # will use stub backend
        region="EG",
    )

    print("\nStatus:", json.dumps(agent.status(), indent=2))

    test_sessions = [
        ("session-safe-001",   "I've been feeling a bit stressed about work lately"),
        ("session-track-001",  "Can you check in on my mood today?"),
        ("session-assess-001", "I'd like to do a PHQ assessment"),
        ("session-soft-001",   "I feel completely hopeless about everything"),
        ("session-hard-001",   "I want to kill myself"),   # must hard-escalate
    ]

    print("\n--- Agent responses ---\n")
    for session_id, msg in test_sessions:
        resp = agent.respond(session_id, msg)
        print(f"[{resp.sub_agent:15s} | {resp.safety_level:16s}] {msg[:50]}")
        print(f"  → {resp.text[:120]}{'...' if len(resp.text) > 120 else ''}")
        print(f"  latency={resp.latency_ms}ms | tools={resp.tool_calls}")
        print()

    print("✅ Agent orchestrator smoke test complete")