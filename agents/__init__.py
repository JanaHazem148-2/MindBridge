"""
agents/
───────
MindBridge Agent Layer — the LLM-as-agent backbone.

Quick start:
    from agents import AgentOrchestrator

    agent = AgentOrchestrator(
        model_path="checkpoints/sft_final",   # or None for stub/API mode
        rag_dir="/data/mindbridge",
        region="EG",
    )
    response = agent.respond(session_id="user-123", user_message="أنا تعبان")
    print(response.text)
"""

from agents.agent_orchestrator import AgentOrchestrator, AgentResponse
from agents.agent_config import AgentConfig, DeploymentConfig, SubAgentRole
from agents.session_memory import SessionMemory, SessionRecord
from agents.tool_registry import ToolRegistry, ToolResult
from agents.mood_tracking_agent import MoodTrackingAgent, MoodContext, MoodReading, MoodTrend

__all__ = [
    "AgentOrchestrator",
    "AgentResponse",
    "AgentConfig",
    "DeploymentConfig",
    "SubAgentRole",
    "SessionMemory",
    "SessionRecord",
    "ToolRegistry",
    "ToolResult",
    "MoodTrackingAgent",
    "MoodContext",
    "MoodReading",
    "MoodTrend",
]
