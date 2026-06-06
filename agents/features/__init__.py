"""
agents/features/
─────────────────
Phase 5 special feature agents.
Build after core agents are stable.

    from agents.features.mood_mirror import MoodMirrorAgent, EmotionSignals
    from agents.features.ifs_parts_map import IFSOrchestrator
    from agents.features.ritual_planner import RitualPlannerAgent
    from agents.features.transference_simulator import TransferenceSimulator
"""
from agents.features.mood_mirror           import MoodMirrorAgent, EmotionSignals, ToneAdjustment
from agents.features.ifs_parts_map         import IFSOrchestrator, InnerPart
from agents.features.ritual_planner        import RitualPlannerAgent, RitualRecommendation
from agents.features.transference_simulator import TransferenceSimulator, SimulationResult

__all__ = [
    "MoodMirrorAgent", "EmotionSignals", "ToneAdjustment",
    "IFSOrchestrator", "InnerPart",
    "RitualPlannerAgent", "RitualRecommendation",
    "TransferenceSimulator", "SimulationResult",
]
