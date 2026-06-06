"""
MindBridge Flask API Server
Connects the full MindBridge backend to the web frontend.
"""
import os, sys, uuid, json, logging, time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("mindbridge.server")

app = Flask(__name__, static_folder="static", static_url_path="")

# Manual CORS headers
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

# ── Shared data directory ─────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_DB = str(DATA_DIR / "users.db")

# ── Build the agent (lazy singleton) ─────────────────────────────────────────
_agent  = None
_bridge = None

def get_agent():
    global _agent
    if _agent is None:
        import config
        try:
            config.validate()
        except EnvironmentError as e:
            log.error(f"Config error: {e}")
        from agents.agent_orchestrator import AgentOrchestrator
        _agent = AgentOrchestrator(
            model_path=None,
            rag_dir=config.RAG_DIR or None,
            memory_dir=str(DATA_DIR),   # ← shared with SessionBridge
            region=config.REGION,
        )
        log.info(f"AgentOrchestrator ready → {DATA_DIR}")
    return _agent

def get_bridge():
    global _bridge
    if _bridge is None:
        from auth.Session_bridge import SessionBridge
        _bridge = SessionBridge(
            memory_dir=str(DATA_DIR),
            users_db_path=USERS_DB,
        )
        log.info(f"SessionBridge → {DATA_DIR}")
    return _bridge

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    import re
    data       = request.get_json(force=True)
    message    = data.get("message", "").strip()
    session_id = data.get("session_id") or f"web-{uuid.uuid4().hex[:12]}"
    is_arabic  = bool(re.search(r'[\u0600-\u06FF]', message))

    if not message:
        return jsonify({"error": "message required"}), 400

    try:
        agent = get_agent()
        resp  = agent.respond(session_id=session_id, user_message=message)

        # Persist turn to bridge (non-fatal)
        try:
            from auth.Session_bridge import SessionBridge
            bridge = get_bridge()
            bridge.add_turn(
                session_id=session_id,
                user=message,
                assistant=resp.text,
                metadata={"sub_agent": resp.sub_agent, "safety_level": resp.safety_level},
            )
        except Exception as be:
            log.warning(f"add_turn non-fatal: {be}")

        return jsonify({
            "reply":        resp.text,
            "session_id":   resp.session_id,
            "sub_agent":    resp.sub_agent,
            "safety_level": resp.safety_level,
            "latency_ms":   resp.latency_ms,
            "tool_calls":   resp.tool_calls,
            "is_arabic":    is_arabic,
        })
    except Exception as e:
        err = str(e)
        log.exception(f"Chat error: {err}")
        # Give a meaningful error message based on cause
        if "GROQ_API_KEY" in err or "api_key" in err.lower():
            detail = "GROQ_API_KEY غير مضبوط — أضفه في ملف .env" if is_arabic else "GROQ_API_KEY not configured — add it to your .env file"
        elif "rate" in err.lower() or "quota" in err.lower():
            detail = "تجاوزت حد الطلبات — انتظر دقيقة" if is_arabic else "Rate limit reached — please wait a moment"
        elif "connection" in err.lower() or "timeout" in err.lower():
            detail = "تعذّر الاتصال بالخادم" if is_arabic else "Connection error — check your internet"
        else:
            detail = f"خطأ: {err[:150]}" if is_arabic else f"Error: {err[:150]}"
        fb = "عذراً، حدث خطأ تقني." if is_arabic else "Technical issue. Please try again."
        return jsonify({"error": detail, "reply": fb}), 500

@app.route("/api/session/new", methods=["POST"])
def new_session():
    session_id = f"web-{uuid.uuid4().hex[:12]}"
    return jsonify({"session_id": session_id})

@app.route("/api/session/<session_id>/progress", methods=["GET"])
def get_progress(session_id):
    try:
        agent = get_agent()
        record = agent.memory.get_session(session_id)
        if not record:
            return jsonify({"session_id": session_id, "total_turns": 0, "phq_history": [], "mood_history": [], "active_goals": [], "summary": None})
        return jsonify(agent.memory.get_progress_report(session_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/mood", methods=["POST"])
def log_mood():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    mood   = int(data.get("mood",   5))
    energy = int(data.get("energy", 5))
    sleep  = int(data.get("sleep",  5))
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    try:
        agent = get_agent()
        agent.memory.log_mood(session_id, mood, energy, sleep)
        return jsonify({"ok": True, "mood": mood, "energy": energy, "sleep": sleep})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ritual", methods=["POST"])
def get_ritual():
    data = request.get_json(force=True)
    session_id  = data.get("session_id", "web-default")
    mood_score  = int(data.get("mood_score", 5))
    user_context = data.get("context", "feeling stressed")
    try:
        agent = get_agent()
        from agents.features.ritual_planner import RitualPlannerAgent
        planner = RitualPlannerAgent(llm=agent.llm)
        rec = planner.recommend(session_id, mood_score, user_context)
        return jsonify({
            "ritual_type":  rec.ritual_type,
            "ritual_name":  rec.ritual_name,
            "duration_min": rec.duration_min,
            "why":          rec.why,
            "steps":        rec.steps,
            "after_prompt": rec.after_prompt,
            "schedule_time": rec.schedule_time,
        })
    except Exception as e:
        log.exception("Ritual error")
        return jsonify({"error": str(e)}), 500

@app.route("/api/status", methods=["GET"])
def status():
    try:
        import config
        agent = get_agent()
        s = agent.status()
        return jsonify({**s, "model": config.GROQ_MODEL, "region": config.REGION})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": time.time()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    log.info(f"Starting MindBridge server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)