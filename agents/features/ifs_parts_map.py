"""
agents/features/ifs_parts_map.py
──────────────────────────────────
Phase 5 — Internal Family Systems (IFS) Parts Map Agent.

Roadmap: "IFS Parts Map (sub-agents per inner part) — build after core stable."

This is a real multi-agent system, not a single prompt.

Architecture:
  Each "inner part" the user identifies becomes a tracked Part object.
  A PartAgent manages each part's voice, positive intention, and fears.
  The SelfFacilitator (lead agent) moderates the dialogue between parts.
  The IFSOrchestrator coordinates everything and routes turns.

IFS Core Concepts implemented:
  - Parts: inner sub-personalities (Critic, Protector, Exile, Manager, Firefighter)
  - Self: the calm, compassionate centre — what we're cultivating
  - Unburdening: when a part releases its extreme role
  - Blending: when a part takes over (vs. Self staying separate)
  - Direct access: speaking TO a part, not FROM it

Usage:
    ifs = IFSOrchestrator(llm=llm_orchestrator)

    # Identify parts from user message:
    parts = ifs.identify_parts("Part of me wants to quit, but another part is terrified.")

    # Start a moderated dialogue:
    response = ifs.facilitate(session_id, user_message)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ── Part definition ───────────────────────────────────────────────────────────

@dataclass
class InnerPart:
    part_id:           str
    name:              str             # e.g. "The Critic", "The Protector"
    part_type:         str             # manager | firefighter | exile | self
    positive_intention: str            # what does this part WANT for the person?
    fears:             List[str]       # what is this part afraid of?
    current_burden:    Optional[str]   # what burden is this part carrying?
    first_appeared:    str             # when did user first mention this part?
    activation_count:  int = 0         # how often has this part been active?
    unburdened:        bool = False    # has this part been unburdened?
    voice_style:       str = ""        # how this part speaks (for LLM prompting)


@dataclass
class PartsMapSession:
    session_id:   str
    parts:        List[InnerPart] = field(default_factory=list)
    self_access:  float = 0.0     # 0-1 estimate of how much Self is present
    active_part:  Optional[str] = None    # part_id currently speaking
    turn_count:   int = 0
    last_updated: float = field(default_factory=time.time)


# ── Part type templates ───────────────────────────────────────────────────────

PART_TYPE_TEMPLATES = {
    "manager": {
        "description":  "Tries to keep things under control; often critical or perfectionist",
        "voice_style":  "direct, controlling, often critical or dismissive of emotions",
        "common_fears": ["losing control", "being overwhelmed", "being seen as weak"],
    },
    "firefighter": {
        "description":  "Acts impulsively to douse emotional pain (e.g. via avoidance, substances, anger)",
        "voice_style":  "urgent, reactive, seeking immediate relief",
        "common_fears": ["being consumed by pain", "exile parts being exposed"],
    },
    "exile": {
        "description":  "Carries pain, shame, or trauma — often a younger part",
        "voice_style":  "small, sad, longing, often fearful or ashamed",
        "common_fears": ["being abandoned", "being too much", "never being seen"],
    },
    "self": {
        "description":  "The calm, compassionate, curious centre",
        "voice_style":  "curious, warm, patient, non-reactive",
        "common_fears": [],
    },
}

# Common parts language patterns to detect in user messages
PART_DETECTION_PATTERNS = {
    "manager":     ["part of me", "inner critic", "should", "must", "have to", "I keep telling myself"],
    "firefighter": ["just want to escape", "numb out", "blow up", "can't stop", "urge to"],
    "exile":       ["little me", "young me", "feels like a child", "so scared", "ashamed", "alone"],
}


# ── Part Agent ────────────────────────────────────────────────────────────────

class PartAgent:
    """
    Manages a single inner part's voice in the IFS dialogue.
    Each part has its own system prompt and perspective.
    """

    def __init__(self, part: InnerPart, llm):
        self.part = part
        self.llm  = llm

    def speak(self, facilitator_prompt: str) -> str:
        """Generate what this part would say in the dialogue."""
        system = (
            f"You are '{self.part.name}', an inner part in an IFS therapy session.\n\n"
            f"Your type: {self.part.part_type}\n"
            f"Your positive intention: {self.part.positive_intention}\n"
            f"Your fears: {'; '.join(self.part.fears)}\n"
            f"Your voice: {self.part.voice_style or PART_TYPE_TEMPLATES.get(self.part.part_type, {}).get('voice_style', 'authentic')}\n\n"
            "You are being spoken to directly by the person's Self (their compassionate centre).\n"
            "Speak as this part — authentically, from its perspective.\n"
            "Keep your response to 2-4 sentences. Be genuine, not theatrical.\n"
            "This is a therapy exercise, not a performance."
        )
        return self.llm.generate(
            system_prompt=system,
            conversation=[{"role": "user", "content": facilitator_prompt}],
        )

    def respond_to_care(self, care_message: str) -> str:
        """How this part responds when the Self offers it compassion."""
        system = (
            f"You are '{self.part.name}' in an IFS session.\n"
            "The person's Self is offering you compassion and understanding.\n"
            "How do you respond? What shifts in you when you feel truly seen?\n"
            "Keep it short (2-3 sentences). Authentic, not dramatic."
        )
        return self.llm.generate(
            system_prompt=system,
            conversation=[{"role": "user", "content": care_message}],
        )


# ── Self Facilitator ──────────────────────────────────────────────────────────

class SelfFacilitator:
    """
    Moderates the IFS dialogue from the position of the user's Self.
    Helps the person speak TO their parts, not FROM them.
    """

    def __init__(self, llm):
        self.llm = llm

    def introduce_part(self, part: InnerPart, parts_context: str) -> str:
        system = (
            "You are a MindBridge IFS Facilitator. "
            "Your role is to help the person's calm Self connect with an inner part.\n\n"
            "Guidelines:\n"
            "- Speak to the person (second person: 'you')\n"
            "- Help them notice the part with curiosity, not judgment\n"
            "- Name the part's positive intention early — every part has one\n"
            "- Keep the person's Self separate from the part (don't let them blend)\n"
            "- Use simple, accessible language — this isn't an IFS textbook\n\n"
            f"Parts active in this session:\n{parts_context}"
        )
        return self.llm.generate(
            system_prompt=system,
            conversation=[{"role": "user", "content": f"Help me work with '{part.name}'."}],
        )

    def facilitate_dialogue(
        self,
        user_message: str,
        active_part: InnerPart,
        parts_context: str,
        session_history: List[Dict],
    ) -> str:
        system = (
            "You are the MindBridge IFS Facilitator moderating an internal dialogue.\n\n"
            f"Parts active in this session:\n{parts_context}\n\n"
            f"Part currently speaking: {active_part.name} ({active_part.part_type})\n"
            f"Positive intention of this part: {active_part.positive_intention}\n\n"
            "Your role:\n"
            "1. Identify which part seems to be speaking in the user's message\n"
            "2. Validate that part's positive intention (every part wants to help)\n"
            "3. Help the user's Self witness the part without merging with it\n"
            "4. If the user is blended (speaking FROM the part), gently unblend\n"
            "5. Guide towards understanding, not fixing\n\n"
            "Tone: curious, warm, unhurried. Never pathologising.\n"
            "Length: 3-5 sentences. This is dialogue, not a lecture."
        )
        return self.llm.generate(
            system_prompt=system,
            conversation=session_history[-6:] + [{"role": "user", "content": user_message}],
        )

    def offer_unburdening(self, part: InnerPart) -> str:
        """Guide an unburdening sequence when a part is ready."""
        system = (
            "You are a MindBridge IFS Facilitator guiding an unburdening sequence.\n\n"
            f"Part being unburdened: {part.name}\n"
            f"Burden it has been carrying: {part.current_burden or 'their protective role'}\n"
            f"Positive intention: {part.positive_intention}\n\n"
            "Guide the person through:\n"
            "1. Acknowledging the part's long service\n"
            "2. Thanking the part for protecting them\n"
            "3. Asking the part if it's ready to release the burden\n"
            "4. A simple ritual to release (visualisation or gesture)\n\n"
            "Keep it gentle and unhurried. This is sacred work."
        )
        return self.llm.generate(
            system_prompt=system,
            conversation=[{"role": "user", "content": "I'm ready to work with this part."}],
        )


# ── IFS Orchestrator ──────────────────────────────────────────────────────────

class IFSOrchestrator:
    """
    Top-level IFS coordinator.
    Manages part identification, dialogue routing, and unburdening.
    """

    def __init__(self, llm):
        self.llm        = llm
        self.facilitator = SelfFacilitator(llm)
        self._sessions:  Dict[str, PartsMapSession] = {}
        self._part_agents: Dict[str, PartAgent] = {}   # part_id → PartAgent

    def _get_or_create_session(self, session_id: str) -> PartsMapSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = PartsMapSession(session_id=session_id)
        return self._sessions[session_id]

    def identify_parts(self, user_message: str, session_id: str) -> List[InnerPart]:
        """
        Analyse a user message and identify inner parts.
        Returns newly identified parts (not already tracked).
        """
        system = (
            "You are an IFS-trained analyst. Extract inner parts from the user's message.\n\n"
            "For each distinct inner part identified, return a JSON object:\n"
            '{"name": "...", "type": "manager|firefighter|exile|self", '
            '"positive_intention": "...", "fears": ["..."], "voice_sample": "..."}\n\n'
            "Return a JSON array. If no distinct parts are identifiable, return []. "
            "Return ONLY valid JSON, no markdown."
        )
        raw = self.llm.generate(
            system_prompt=system,
            conversation=[{"role": "user", "content": user_message}],
        )
        parts_identified = []
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data  = json.loads(clean)
            for item in data:
                part = InnerPart(
                    part_id=str(uuid.uuid4())[:8],
                    name=item.get("name", "Unnamed Part"),
                    part_type=item.get("type", "manager"),
                    positive_intention=item.get("positive_intention", "to keep you safe"),
                    fears=item.get("fears", []),
                    current_burden=None,
                    first_appeared=user_message[:80],
                    voice_style=item.get("voice_sample", ""),
                )
                parts_identified.append(part)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Part identification parse error: {e} — returning empty")
        return parts_identified

    def add_parts(self, session_id: str, parts: List[InnerPart]):
        """Register newly identified parts into the session."""
        session = self._get_or_create_session(session_id)
        for part in parts:
            # Avoid duplicates by name
            existing_names = {p.name for p in session.parts}
            if part.name not in existing_names:
                session.parts.append(part)
                self._part_agents[part.part_id] = PartAgent(part, self.llm)

    def facilitate(
        self,
        session_id: str,
        user_message: str,
        session_history: Optional[List[Dict]] = None,
        auto_identify: bool = True,
    ) -> str:
        """
        Main entry point — facilitate one IFS turn.
        Auto-identifies new parts if auto_identify=True.
        """
        session = self._get_or_create_session(session_id)
        session.turn_count += 1

        # Auto-identify new parts
        if auto_identify:
            new_parts = self.identify_parts(user_message, session_id)
            if new_parts:
                self.add_parts(session_id, new_parts)
                logger.info(f"IFS: {len(new_parts)} new part(s) identified: {[p.name for p in new_parts]}")

        # Build parts context for the facilitator
        parts_context = self._build_parts_context(session)

        if not session.parts:
            # No parts identified yet — use general IFS opening
            return self.llm.generate(
                system_prompt=(
                    "You are a MindBridge IFS Facilitator. The user hasn't clearly named any inner parts yet. "
                    "Gently introduce the concept of inner parts and invite exploration. "
                    "Be warm, curious, and non-clinical. 2-4 sentences."
                ),
                conversation=[(session_history or []) + [{"role": "user", "content": user_message}]][-1:],
            )

        # Determine which part is active
        active_part = self._identify_active_part(user_message, session)

        return self.facilitator.facilitate_dialogue(
            user_message=user_message,
            active_part=active_part,
            parts_context=parts_context,
            session_history=session_history or [],
        )

    def _build_parts_context(self, session: PartsMapSession) -> str:
        if not session.parts:
            return "No parts identified yet."
        lines = []
        for p in session.parts:
            status = "✓ unburdened" if p.unburdened else f"active ({p.activation_count}x)"
            lines.append(
                f"• {p.name} ({p.part_type}) — intention: {p.positive_intention} [{status}]"
            )
        return "\n".join(lines)

    def _identify_active_part(self, user_message: str, session: PartsMapSession) -> InnerPart:
        """Simple heuristic to identify which part is currently speaking."""
        msg_lower = user_message.lower()
        # Look for name mentions
        for part in session.parts:
            if part.name.lower() in msg_lower:
                part.activation_count += 1
                return part
        # Default to most recently active, or first part
        return session.parts[-1]

    def get_parts_map(self, session_id: str) -> Dict:
        """Return a structured parts map for the clinician dashboard."""
        session = self._get_or_create_session(session_id)
        return {
            "session_id": session_id,
            "turn_count": session.turn_count,
            "self_access": session.self_access,
            "parts": [
                {
                    "id":          p.part_id,
                    "name":        p.name,
                    "type":        p.part_type,
                    "intention":   p.positive_intention,
                    "fears":       p.fears,
                    "activations": p.activation_count,
                    "unburdened":  p.unburdened,
                }
                for p in session.parts
            ],
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== IFSOrchestrator — Smoke Test (stub LLM) ===\n")

    class StubLLM:
        def generate(self, system_prompt, conversation, **kw):
            return "[STUB] IFS response — add GROQ_API_KEY to test with real LLM"

    ifs = IFSOrchestrator(llm=StubLLM())
    sid = "ifs-test-001"

    messages = [
        "Part of me wants to open up but another part is screaming I'm pathetic for trying.",
        "The critical voice is really loud today. I can't seem to quiet it.",
        "I think the scared little part is underneath all this anger.",
    ]
    for msg in messages:
        print(f"User: {msg}")
        resp = ifs.facilitate(sid, msg)
        print(f"IFS : {resp[:150]}\n")

    import json
    print("Parts map:")
    print(json.dumps(ifs.get_parts_map(sid), indent=2))
    print("\n✅ IFSOrchestrator smoke test complete")
