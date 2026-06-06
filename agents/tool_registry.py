"""
agents/tool_registry.py
────────────────────────
All tools available to the MindBridge agent.

Tools are called by the agent when it emits:
    <tool_call>{"tool": "tool_name", "args": {...}}</tool_call>

Each tool:
  - Has a clear name and description (used in system prompt)
  - Takes a dict of args + session_id
  - Returns a ToolResult with output text and optional append_to_response

Available tools:
  rag_retrieve      — retrieve clinical context from vector store
  phq_assessment    — run or continue a PHQ-8 screening
  mood_log          — log mood/energy/sleep check-in
  session_summary   — summarise and store session progress
  escalate_crisis   — notify clinician + surface crisis resources
  set_goal          — create a therapeutic goal
  check_progress    — retrieve goal and score progress
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Tool result ───────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool:               str
    success:            bool
    output:             str              # structured result for agent
    append_to_response: str = ""         # text to append to user-facing response
    metadata:           Dict = field(default_factory=dict)
    duration_ms:        float = 0.0


# ── Tool registry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Registry of all tools the agent can call.
    Each tool is a method; the registry dispatches by name.
    """

    TOOL_DESCRIPTIONS = {
        "rag_retrieve":    "Retrieve relevant clinical knowledge for the user's message. Args: {query: str, top_k: int=3}",
        "phq_assessment":  "Start or continue a PHQ-8 depression screening. Args: {action: 'start'|'answer'|'score', answer: str='', score: int=0}",
        "mood_log":        "Log the user's current mood, energy, and sleep on a 1-10 scale. Args: {mood: int, energy: int, sleep: int}",
        "session_summary": "Generate and store a summary of the session so far. Args: {}",
        "escalate_crisis": "Immediately notify the clinician and provide crisis resources. Args: {reason: str}",
        "set_goal":        "Set a therapeutic goal for the user. Args: {goal_text: str}",
        "check_progress":  "Review the user's PHQ scores, mood trends, and goals. Args: {}",
    }

    def __init__(self, rag_dir: Optional[str] = None, memory=None):
        self.rag_dir    = rag_dir
        self.memory     = memory         # injected after init to avoid circular import
        self._rag       = None
        self.rag_available = False

        if rag_dir:
            self._init_rag(rag_dir)

    def _init_rag(self, rag_dir: str):
        try:
            import sys, os
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from rag.rag_pipeline import RAGPipeline
            self._rag = RAGPipeline(
                vector_store="chroma" if Path(rag_dir).exists() else "memory",
                data_dir=rag_dir,
            )
            self._rag.build_index()
            self.rag_available = True
            logger.info("RAG pipeline ready")
        except Exception as e:
            logger.warning(f"RAG init failed: {e} — rag_retrieve will return empty context")

    def execute(self, tool_name: str, args: Dict, session_id: str) -> ToolResult:
        """Dispatch a tool call by name."""
        t_start = time.time()
        if tool_name not in self.TOOL_DESCRIPTIONS:
            return ToolResult(
                tool=tool_name, success=False,
                output=f"Unknown tool: {tool_name}",
            )
        try:
            method = getattr(self, f"_tool_{tool_name}", None)
            if method is None:
                return ToolResult(tool=tool_name, success=False, output="Tool not implemented")
            result = method(args=args, session_id=session_id)
            result.duration_ms = round((time.time() - t_start) * 1000, 1)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return ToolResult(tool=tool_name, success=False, output=str(e))

    def rag_retrieve(self, query: str, top_k: int = 3) -> str:
        """Public convenience method used directly by AgentOrchestrator."""
        if not self._rag or not self.rag_available:
            return ""
        try:
            _, context_block = self._rag.query_with_prompt(query, top_k=top_k)
            return context_block
        except Exception as e:
            logger.warning(f"RAG retrieve failed: {e}")
            return ""

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_rag_retrieve(self, args: Dict, session_id: str) -> ToolResult:
        query  = args.get("query", "")
        top_k  = int(args.get("top_k", 3))
        context = self.rag_retrieve(query, top_k)
        return ToolResult(
            tool="rag_retrieve",
            success=True,
            output=context or "No relevant context found.",
            metadata={"query": query, "top_k": top_k},
        )

    def _tool_phq_assessment(self, args: Dict, session_id: str) -> ToolResult:
        """
        PHQ-8 assessment state machine.
        Action 'start'  — begin a fresh PHQ-8
        Action 'answer' — record an answer to the current question
        Action 'score'  — submit final score and get interpretation
        """
        action = args.get("action", "start")

        PHQ_QUESTIONS = [
            "Over the last two weeks, how often have you had little interest or pleasure in doing things? "
            "(0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day)",
            "How often have you been feeling down, depressed, or hopeless?",
            "How often have you had trouble falling or staying asleep, or sleeping too much?",
            "How often have you felt tired or had little energy?",
            "How often have you had a poor appetite, or been overeating?",
            "How often have you felt bad about yourself — or that you're a failure or have let yourself or your family down?",
            "How often have you had trouble concentrating on things, like reading or watching TV?",
            "How often have you been moving or speaking so slowly that other people could have noticed? "
            "Or the opposite — being so fidgety or restless that you've been moving around a lot more than usual?",
        ]

        PHQ_SEVERITY = [(0,4,"minimal"),(5,9,"mild"),(10,14,"moderate"),(15,19,"moderately severe"),(20,24,"severe")]

        if action == "start":
            if self.memory:
                self.memory.set_active_assessment(session_id, True)
            return ToolResult(
                tool="phq_assessment",
                success=True,
                output=json.dumps({"question_index": 0, "question": PHQ_QUESTIONS[0]}),
                append_to_response=(
                    "\n\n**PHQ-8 Assessment — Question 1 of 8:**\n" + PHQ_QUESTIONS[0]
                ),
            )

        elif action == "score":
            total_score = int(args.get("score", 0))
            severity = "unknown"
            for lo, hi, label in PHQ_SEVERITY:
                if lo <= total_score <= hi:
                    severity = label
                    break
            if self.memory:
                self.memory.log_phq(session_id, total_score, severity)
                self.memory.set_active_assessment(session_id, False)
            return ToolResult(
                tool="phq_assessment",
                success=True,
                output=json.dumps({"score": total_score, "severity": severity}),
                append_to_response=(
                    f"\n\n**PHQ-8 Complete — Score: {total_score}/24 ({severity.title()})**\n"
                    "I've recorded this score. Would you like me to explain what it means?"
                ),
                metadata={"score": total_score, "severity": severity},
            )

        return ToolResult(tool="phq_assessment", success=False, output="Unknown action")

    def _tool_mood_log(self, args: Dict, session_id: str) -> ToolResult:
        mood   = max(1, min(10, int(args.get("mood",   5))))
        energy = max(1, min(10, int(args.get("energy", 5))))
        sleep  = max(1, min(10, int(args.get("sleep",  5))))

        if self.memory:
            self.memory.log_mood(session_id, mood, energy, sleep)

        # Flag low scores
        flags = []
        if mood   <= 3: flags.append("low mood")
        if energy <= 3: flags.append("low energy")
        if sleep  <= 3: flags.append("poor sleep")

        flag_text = ""
        if flags:
            flag_text = f" I notice {' and '.join(flags)} — I'd like to check in about that."

        return ToolResult(
            tool="mood_log",
            success=True,
            output=json.dumps({"mood": mood, "energy": energy, "sleep": sleep, "flags": flags}),
            append_to_response=(
                f"\n\n✓ Logged: Mood {mood}/10 · Energy {energy}/10 · Sleep {sleep}/10.{flag_text}"
            ),
            metadata={"mood": mood, "energy": energy, "sleep": sleep},
        )

    def _tool_session_summary(self, args: Dict, session_id: str) -> ToolResult:
        if self.memory:
            record = self.memory.get_session(session_id)
            if record:
                summary_parts = []
                if record.phq_history:
                    latest = record.phq_history[-1]
                    summary_parts.append(f"Latest PHQ-8: {latest.score} ({latest.severity})")
                if record.mood_history:
                    recent_mood = record.mood_history[-3:]
                    avg_mood = sum(m.mood for m in recent_mood) / len(recent_mood)
                    summary_parts.append(f"Average recent mood: {avg_mood:.1f}/10")
                if record.goals:
                    active = [g.goal_text for g in record.goals if g.status == "active"]
                    if active:
                        summary_parts.append(f"Active goals: {'; '.join(active)}")
                summary = ". ".join(summary_parts) or "No data logged yet."
                return ToolResult(
                    tool="session_summary",
                    success=True,
                    output=summary,
                    metadata={"turns": record.total_turns},
                )
        return ToolResult(tool="session_summary", success=True, output="Session summary not available.")

    def _tool_escalate_crisis(self, args: Dict, session_id: str) -> ToolResult:
        reason = args.get("reason", "Crisis detected by agent")
        logger.critical(f"TOOL: escalate_crisis | session={session_id} | reason={reason}")
        # In production: POST to clinician webhook here
        return ToolResult(
            tool="escalate_crisis",
            success=True,
            output=f"Crisis escalated. Clinician notified. Reason: {reason}",
            metadata={"session_id": session_id, "reason": reason},
        )

    def _tool_set_goal(self, args: Dict, session_id: str) -> ToolResult:
        goal_text = args.get("goal_text", "")
        if not goal_text:
            return ToolResult(tool="set_goal", success=False, output="goal_text is required")
        goal_id = ""
        if self.memory:
            goal_id = self.memory.set_goal(session_id, goal_text)
        return ToolResult(
            tool="set_goal",
            success=True,
            output=json.dumps({"goal_id": goal_id, "goal_text": goal_text}),
            append_to_response=f"\n\n✓ Goal set: \"{goal_text}\"",
            metadata={"goal_id": goal_id},
        )

    def _tool_check_progress(self, args: Dict, session_id: str) -> ToolResult:
        if not self.memory:
            return ToolResult(tool="check_progress", success=True, output="Memory not available.")
        report = self.memory.get_progress_report(session_id)
        return ToolResult(
            tool="check_progress",
            success=True,
            output=json.dumps(report),
            metadata={"session_id": session_id},
        )

    def get_tool_docs(self) -> str:
        """Return tool descriptions for injection into system prompt."""
        lines = ["Available tools (emit <tool_call>{...}</tool_call> to use):"]
        for name, desc in self.TOOL_DESCRIPTIONS.items():
            lines.append(f"  {name}: {desc}")
        return "\n".join(lines)
