"""
main.py
────────
MindBridge — main entry point.

Sends a user message through the full pipeline:
  SafetyOrchestrator → AgentOrchestrator (router + memory + RAG)
    → InferenceBridge → OpenAI gpt-4o → response

Usage:
    python main.py
    python main.py --message "I've been feeling anxious"
    python main.py --session my-session-123 --message "..."
    python main.py --interactive
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_agent():
    """Construct the AgentOrchestrator wired to the OpenAI backend."""
    from agents.agent_orchestrator import AgentOrchestrator
    from llm.inference_bridge import InferenceBridge

    # Build the OpenAI-backed LLM bridge
    llm_bridge = InferenceBridge(
        use_openai   = True,
        max_new_tokens = config.MAX_TOKENS,
        temperature  = config.TEMPERATURE,
        top_p        = config.TOP_P,
        fallback_to_stub = True,
    )

    # Construct the agent, injecting our bridge as the LLM backend
    agent = AgentOrchestrator(
        model_path = None,       # not using local weights
        rag_dir    = config.RAG_DIR or None,
        memory_dir = config.MEMORY_DIR,
        region     = config.REGION,
    )

    # Swap the default LLM with our OpenAI bridge
    agent.llm = llm_bridge

    logger.info(
        f"Agent ready | backend={llm_bridge.backend} | "
        f"model={llm_bridge.model} | region={config.REGION}"
    )
    return agent


def run_single(agent, session_id: str, message: str) -> None:
    """Send one message and print the response."""
    print(f"\n{'─'*60}")
    print(f"Session : {session_id}")
    print(f"User    : {message}")
    print(f"{'─'*60}")

    response = agent.respond(session_id=session_id, user_message=message)

    print(f"Agent   [{response.sub_agent}] [{response.safety_level}]")
    print(f"\n{response.text}\n")
    print(f"Latency : {response.latency_ms}ms")
    if response.tool_calls:
        print(f"Tools   : {[tc.get('tool', '?') for tc in response.tool_calls]}")
    print(f"{'─'*60}\n")


def run_interactive(agent) -> None:
    """Simple REPL for interactive testing."""
    session_id = f"interactive-{uuid.uuid4().hex[:8]}"
    print(f"\nMindBridge Interactive Session  (session: {session_id})")
    print("Type 'quit' or Ctrl-C to exit.\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not message:
            continue
        if message.lower() in ("quit", "exit", "q"):
            print("Session ended.")
            break

        response = agent.respond(session_id=session_id, user_message=message)
        print(f"\nMindBridge [{response.sub_agent}]: {response.text}\n")


def main():
    parser = argparse.ArgumentParser(description="MindBridge — AI mental health companion")
    parser.add_argument("--message",     "-m", default=None,  help="Single message to send")
    parser.add_argument("--session",     "-s", default=None,  help="Session ID (auto-generated if omitted)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--status",            action="store_true", help="Print config status and exit")
    args = parser.parse_args()

    # Config status
    if args.status:
        print(json.dumps(config.summary(), indent=2))
        return

    # Validate API key
    try:
        config.validate()
    except EnvironmentError as e:
        print(f"\n⚠️  {e}")
        sys.exit(1)

    agent = build_agent()

    if args.interactive:
        run_interactive(agent)
        return

    # Single message (or default demo)
    session_id = args.session or f"demo-{uuid.uuid4().hex[:8]}"
    message    = args.message or "I've been feeling really anxious and overwhelmed lately."
    run_single(agent, session_id, message)


if __name__ == "__main__":
    main()
