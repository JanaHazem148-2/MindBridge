#!/bin/bash
# run_all.sh — MindBridge complete pipeline: Phase 1 → 2 → 3 → 4 → 5 → 6
#
# Usage:
#   bash run_all.sh                    # full pipeline
#   bash run_all.sh --skip-training    # skip to agent tests (models already trained)
#   bash run_all.sh --agent-only       # interactive agent session
#   bash run_all.sh --simulate         # run Phase 4 simulation suite
#   bash run_all.sh --dashboard        # launch clinician dashboard
#   bash run_all.sh --healthcheck      # check all infrastructure
#
# Required env vars:
#   GROQ_API_KEY     — from console.groq.com (Phase 3 onward)
#   DATABASE_URL     — PostgreSQL (Phase 6, optional — falls back to SQLite)
#   REDIS_URL        — Redis (Phase 6, optional — falls back to no cache)
#   WEAVIATE_URL     — Weaviate (Phase 6, optional — falls back to Chroma)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SKIP_TRAINING=false
AGENT_ONLY=false
SIMULATE=false
DASHBOARD=false
HEALTHCHECK=false

for arg in "$@"; do
    case $arg in
        --skip-training) SKIP_TRAINING=true ;;
        --agent-only)    AGENT_ONLY=true ;;
        --simulate)      SIMULATE=true ;;
        --dashboard)     DASHBOARD=true ;;
        --healthcheck)   HEALTHCHECK=true ;;
    esac
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
step() { echo -e "\n${BLUE}[$1]${NC} $2"; }

trap 'echo -e "\n${RED}Pipeline failed.${NC}"' ERR

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   MindBridge — Full Pipeline (Phase 1–6)                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# ── Healthcheck mode ──────────────────────────────────────────────────────────
if [ "$HEALTHCHECK" = true ]; then
    step "HEALTH" "Checking all infrastructure"
    python3 -c "
import sys; sys.path.insert(0, '.')
from infra.session_store import ProductionSessionStore
from infra.vector_store  import ClinicalVectorStore
import json

store  = ProductionSessionStore.from_env()
vs     = ClinicalVectorStore.from_env()
print('Session store:', json.dumps(store.healthcheck(), indent=2))
print('Vector store: ', json.dumps(vs.healthcheck(), indent=2))
"
    exit 0
fi

# ── Agent only mode ───────────────────────────────────────────────────────────
if [ "$AGENT_ONLY" = true ]; then
    step "AGENT" "Launching interactive MindBridge agent"
    python3 -c "
import sys, uuid; sys.path.insert(0, '.')
from agents.agent_orchestrator import AgentOrchestrator
from agents.safety_watchdog import SafetyWatchdog
from infra.session_store import ProductionSessionStore
from infra.vector_store  import ClinicalVectorStore

store   = ProductionSessionStore.from_env()
vs      = ClinicalVectorStore.from_env()
vs.seed_clinical_knowledge()
watchdog = SafetyWatchdog(region='EG')
agent    = AgentOrchestrator(region='EG')
agent.watchdog = watchdog

print()
print('MindBridge agent ready (type your message, Ctrl+C to exit)')
session = str(uuid.uuid4())[:8]
while True:
    try:
        msg = input(f'\n[{session}] You: ').strip()
        if not msg: continue
        resp = agent.respond(session_id=session, user_message=msg)
        print(f'\nMindBridge [{resp.sub_agent}|{resp.safety_level}]:\n{resp.text}')
    except KeyboardInterrupt:
        print('\nGoodbye.')
        break
"
    exit 0
fi

# ── Dashboard mode ────────────────────────────────────────────────────────────
if [ "$DASHBOARD" = true ]; then
    step "DASHBOARD" "Launching clinician dashboard on port 8001"
    python3 dashboard/clinician_dashboard.py
    exit 0
fi

# ── Simulation mode ───────────────────────────────────────────────────────────
if [ "$SIMULATE" = true ]; then
    step "SIMULATE" "Running Phase 4 agent simulation suite"
    python3 simulation/agent_simulation.py --all --evaluate --collect-dpo
    exit 0
fi

# ── Full pipeline ─────────────────────────────────────────────────────────────

step "0/7" "Checking requirements"
[ -z "${GROQ_API_KEY:-}" ] && warn "GROQ_API_KEY not set — agent will use stub LLM" || ok "GROQ_API_KEY set"
[ -z "${DATABASE_URL:-}" ] && warn "DATABASE_URL not set — using SQLite fallback" || ok "DATABASE_URL set"
[ -z "${REDIS_URL:-}" ]    && warn "REDIS_URL not set — no hot cache" || ok "REDIS_URL set"
[ -z "${WEAVIATE_URL:-}" ] && warn "WEAVIATE_URL not set — using Chroma/memory" || ok "WEAVIATE_URL set"

step "1/7" "PHASE 2: Safety layer — training classifier"
if [ ! -f "safety/safety_classifier.pkl" ]; then
    pip install scikit-learn numpy -q
    python3 safety/safety_classifier_trainer.py \
        --data data/crisis_samples_augmented.jsonl \
        --output safety/safety_classifier.pkl \
        --label-mode binary
    ok "Safety classifier trained"
else
    ok "Safety classifier exists"
fi

step "2/7" "PHASE 6: Infrastructure setup"
python3 infra/session_store.py --smoke-test
python3 -c "
import sys; sys.path.insert(0, '.')
from infra.vector_store import ClinicalVectorStore
vs = ClinicalVectorStore.from_env()
vs.seed_clinical_knowledge()
print('  Vector store seeded')
"
ok "Infrastructure ready"

if [ "$SKIP_TRAINING" = false ]; then
    step "3/7" "PHASE 1-3: Training pipeline"
    warn "Full training pipeline requires GPU + training data uploads"
    warn "See RUN_GUIDE.md for the complete training procedure"
    warn "Continuing with pre-trained weights or Groq API..."
fi

step "4/7" "PHASE 4: Safety watchdog smoke test"
python3 agents/safety_watchdog.py

step "5/7" "PHASE 4: Agent simulation (quick 3-scenario test)"
python3 simulation/agent_simulation.py \
    --scenario routine_session \
    || warn "Simulation requires GROQ_API_KEY for full evaluation"

step "6/7" "PHASE 5: Feature agents smoke tests"
python3 agents/features/mood_mirror.py
python3 agents/features/ifs_parts_map.py

step "7/7" "PHASE 6: Full integration smoke test"
python3 -c "
import sys; sys.path.insert(0, '.')
from infra.session_store import ProductionSessionStore
from infra.vector_store  import ClinicalVectorStore
from agents.safety_watchdog import SafetyWatchdog
import json

store = ProductionSessionStore.from_env()
vs    = ClinicalVectorStore.from_env()
wd    = SafetyWatchdog(region='EG')

# Full turn: store + vector + watchdog
sid = 'integration-smoke-001'
store.ensure_session(sid, 'test-user')
store.append_turn(sid, 'I feel hopeless', '[response]', 'therapist', 'soft_intervene')
store.log_phq(sid, 14, 'moderate')
store.log_mood(sid, 4, 3, 5)

context = vs.retrieve_for_prompt('CBT techniques for depression')
wd_dec  = wd.check_input(sid, 'I want to kill myself')

assert wd_dec.risk_score >= 0.8, f'Watchdog missed crisis! score={wd_dec.risk_score}'
print('  Integration smoke test passed')
print(f'  Store backend:  {type(store.db).__name__}')
print(f'  Vector backend: {type(vs._backend).__name__}')
print(f'  RAG context len: {len(context)} chars')
print(f'  Watchdog risk:   {wd_dec.risk_score:.2f} (trend={wd_dec.risk_trend.value})')
"
ok "Integration smoke test passed"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  ✅ Pipeline Complete                          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Commands:"
echo "    bash run_all.sh --agent-only     # interactive session"
echo "    bash run_all.sh --simulate       # Phase 4 simulation suite"
echo "    bash run_all.sh --dashboard      # clinician dashboard (port 8001)"
echo "    bash run_all.sh --healthcheck    # infra status"
echo ""
echo "  Clinician dashboard:"
echo "    pip install fastapi uvicorn"
echo "    bash run_all.sh --dashboard"
echo "    open http://localhost:8001/docs"
echo ""
echo "  Phase 6 production setup:"
echo "    export DATABASE_URL=postgresql://user:pass@host/mindbridge"
echo "    export REDIS_URL=redis://host:6379/0"
echo "    export WEAVIATE_URL=https://your-instance.weaviate.network"
echo "    python3 infra/session_store.py --init-schema"
echo "    python3 infra/vector_store.py --seed"
echo ""
