"""
agents/agent_config.py
───────────────────────
All tuneable parameters for the MindBridge agent system.
Centralised here so nothing is magic-numbered in agent code.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class SubAgentRole(Enum):
    THERAPIST    = "therapist"
    ASSESSOR     = "assessor"
    MOOD_TRACKER = "mood_tracker"
    CRISIS       = "crisis"
    PLANNER      = "planner"


@dataclass
class AgentConfig:
    # ── LLM generation ────────────────────────────────────────────────────────
    max_new_tokens:  int   = 512
    temperature:     float = 0.7
    top_p:           float = 0.9
    repetition_penalty: float = 1.1     # penalise repeated tokens in therapy text

    # ── Context window ────────────────────────────────────────────────────────
    context_window_turns: int = 6       # how many past turns to include in LLM context
    rag_top_k:            int = 3       # number of RAG documents to retrieve
    max_context_chars:    int = 1200    # max chars of RAG context to inject

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_summary_threshold: int = 10  # summarise after this many turns
    summarise_every_n_turns:  int = 15  # re-summarise every N turns
    max_stored_sessions:      int = 10_000

    # ── Safety ────────────────────────────────────────────────────────────────
    safety_classifier_threshold: float = 0.60  # Layer 2 crisis probability threshold
    hard_escalate_threshold:     float = 0.85  # above this → hard escalate from classifier
    check_output_safety:         bool  = True  # run safety check on every LLM output

    # ── Tool execution ────────────────────────────────────────────────────────
    max_tool_calls_per_turn: int = 3    # prevent runaway tool loops
    tool_timeout_seconds:    int = 5    # max time for any single tool call

    # ── Session ───────────────────────────────────────────────────────────────
    session_timeout_minutes: int = 60   # inactive sessions expire from hot cache
    max_turns_per_session:   int = 200  # safety ceiling

    # ── Logging ───────────────────────────────────────────────────────────────
    log_all_turns:    bool = True
    log_tool_calls:   bool = True
    log_safety_flags: bool = True


@dataclass
class DeploymentConfig:
    """
    Production deployment settings — separate from agent logic config.
    """
    region:            str  = "EG"
    enable_arabic:     bool = True
    clinician_webhook: Optional[str] = None   # POST URL for crisis alerts
    session_store:     str  = "sqlite"        # "sqlite" | "redis" | "memory"
    redis_url:         Optional[str] = None
    sqlite_path:       str  = "/data/mindbridge_sessions.db"
    model_path:        Optional[str] = None
    api_base:          Optional[str] = None
    api_key:           Optional[str] = None
    rag_dir:           Optional[str] = None
    classifier_path:   Optional[str] = None

    @classmethod
    def from_env(cls) -> "DeploymentConfig":
        """Load config from environment variables (12-factor app style)."""
        import os
        return cls(
            region=            os.environ.get("MINDBRIDGE_REGION", "EG"),
            enable_arabic=     os.environ.get("MINDBRIDGE_ARABIC", "1") == "1",
            clinician_webhook= os.environ.get("MINDBRIDGE_WEBHOOK"),
            session_store=     os.environ.get("MINDBRIDGE_SESSION_STORE", "sqlite"),
            redis_url=         os.environ.get("REDIS_URL"),
            sqlite_path=       os.environ.get("MINDBRIDGE_DB_PATH", "/data/mindbridge_sessions.db"),
            model_path=        os.environ.get("MINDBRIDGE_MODEL_PATH"),
            api_base=          os.environ.get("MINDBRIDGE_API_BASE"),
            api_key=           os.environ.get("MINDBRIDGE_API_KEY"),
            rag_dir=           os.environ.get("MINDBRIDGE_RAG_DIR"),
            classifier_path=   os.environ.get("MINDBRIDGE_CLASSIFIER_PATH"),
        )
