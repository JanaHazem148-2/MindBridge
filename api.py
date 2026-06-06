"""
api.py  ─  MindBridge Phase 8 — FastAPI Main Server
══════════════════════════════════════════════════════
Identical to Phase 7 EXCEPT:

PERSISTENCE (Phase 8 addition):
  _alerts and _rlhf_queue are now backed by the production session_store
  (SQLiteBackend by default; swap to PostgresBackend via DATABASE_URL env var;
   optional Redis hot-cache via REDIS_URL env var).

  Alerts table:  stored in SQLite alerts table (see session_store.py schema)
  RLHF queue:    stored in SQLite rlhf_queue table (new table auto-created)
  Sessions:      full chat history / PHQ / mood via ProductionSessionStore

  No data is lost on server restart.

New endpoints (Phase 8):
  GET  /dashboard/phq/{session_id}  — PHQ history for PHQ tab in Clinician.jsx

Environment variables:
  DATABASE_URL   sqlite:///./mindbridge.db  (or postgres://...)
  REDIS_URL      redis://localhost:6379/0   (optional, enables hot-cache)
  JWT_SECRET     (required in production)
  CRISIS_WEBHOOK_URL / SMS_WEBHOOK_URL / PUSH_WEBHOOK_URL

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import uuid
import time
import logging
import asyncio
import json
import sqlite3
import httpx
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import jwt  # PyJWT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

import config

JWT_SECRET   = os.environ.get("JWT_SECRET", "mindbridge-dev-jwt-secret-32chars!!")
JWT_ALGO     = "HS256"
JWT_EXPIRE_H = 24

CRISIS_WEBHOOK_URL = os.environ.get("CRISIS_WEBHOOK_URL", "")
SMS_WEBHOOK_URL    = os.environ.get("SMS_WEBHOOK_URL", "")
PUSH_WEBHOOK_URL   = os.environ.get("PUSH_WEBHOOK_URL", "")

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./mindbridge_api.db")
REDIS_URL    = os.environ.get("REDIS_URL", "")

# ═══════════════════════════════════════════════════════════════════════════════
# Persistent store — thin wrapper around session_store.SQLiteBackend
# ═══════════════════════════════════════════════════════════════════════════════

class PersistentStore:
    """
    Wraps SQLiteBackend (or PostgresBackend) from infra/session_store.py and
    adds two extra tables:
      alerts     — crisis alert queue (survives restart)
      rlhf_queue — RLHF scoring queue (survives restart)

    Falls back to InMemoryFallback if the import fails (dev mode).
    """

    EXTRA_SCHEMA = """
    CREATE TABLE IF NOT EXISTS alerts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      TEXT NOT NULL,
        user_id         TEXT,
        safety_level    TEXT,
        message_snippet TEXT,
        timestamp       REAL NOT NULL,
        acknowledged    INTEGER DEFAULT 0,
        acknowledged_by TEXT,
        acknowledged_at TEXT,
        escalated_at    TEXT
    );

    CREATE TABLE IF NOT EXISTS rlhf_queue (
        id               TEXT PRIMARY KEY,
        session_id       TEXT NOT NULL,
        user_message     TEXT,
        ai_response      TEXT,
        sub_agent        TEXT,
        safety_level     TEXT,
        timestamp        REAL NOT NULL,
        scored           INTEGER DEFAULT 0,
        score            INTEGER,
        clinician_note   TEXT,
        preferred_response TEXT,
        scored_by        TEXT,
        scored_at        TEXT
    );
    """

    def __init__(self):
        # ── Try to use the project's existing SQLiteBackend ───────────────────
        try:
            from infra.session_store import SQLiteBackend, ProductionSessionStore
            db_path = self._resolve_db_path(DATABASE_URL)
            self.db = SQLiteBackend(db_path=db_path)
            logger.info(f"PersistentStore: SQLiteBackend @ {db_path}")
        except ImportError:
            # session_store not importable in this env — use bare sqlite3
            db_path = "./mindbridge_api.db"
            self.db = None
            logger.warning("session_store not importable — using bare sqlite3 fallback")

        # ── Always create our own connection for alerts + rlhf tables ─────────
        self._db_path = db_path if self.db else "./mindbridge_api.db"
        self._ensure_extra_schema()

        # ── Optional Redis cache for alerts (hot unread count) ────────────────
        self._redis = None
        if REDIS_URL:
            try:
                import redis as redis_lib
                self._redis = redis_lib.from_url(REDIS_URL, decode_responses=True)
                self._redis.ping()
                logger.info(f"Redis connected: {REDIS_URL}")
            except Exception as e:
                logger.warning(f"Redis not available ({e}) — continuing without cache")
                self._redis = None

    @staticmethod
    def _resolve_db_path(url: str) -> str:
        if url.startswith("sqlite:///"):
            return url[len("sqlite:///"):]
        return "./mindbridge_api.db"

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_extra_schema(self):
        with self._conn() as conn:
            conn.executescript(self.EXTRA_SCHEMA)
        logger.info("alerts + rlhf_queue tables ensured")

    # ── Alert operations ───────────────────────────────────────────────────────

    def add_alert(self, session_id: str, user_id: str, safety_level: str,
                  message_snippet: str, escalated_at: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO alerts
                   (session_id, user_id, safety_level, message_snippet,
                    timestamp, escalated_at)
                   VALUES (?,?,?,?,?,?)""",
                (session_id, user_id, safety_level, message_snippet,
                 time.time(), escalated_at),
            )
            alert_id = cur.lastrowid

        # Invalidate Redis unread count cache
        if self._redis:
            try:
                self._redis.delete("mb:alerts:unread_count")
            except Exception:
                pass

        return alert_id

    def get_alerts(self, unread_only: bool = False, limit: int = 50) -> List[Dict]:
        with self._conn() as conn:
            if unread_only:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE acknowledged=0 ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    def count_unread_alerts(self) -> int:
        # Try Redis first
        if self._redis:
            try:
                cached = self._redis.get("mb:alerts:unread_count")
                if cached is not None:
                    return int(cached)
            except Exception:
                pass

        with self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE acknowledged=0"
            ).fetchone()[0]

        if self._redis:
            try:
                self._redis.setex("mb:alerts:unread_count", 30, count)  # cache 30 s
            except Exception:
                pass

        return count

    def ack_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            affected = conn.execute(
                """UPDATE alerts
                   SET acknowledged=1, acknowledged_by=?, acknowledged_at=?
                   WHERE id=?""",
                (acknowledged_by, now, alert_id),
            ).rowcount
        if self._redis:
            try:
                self._redis.delete("mb:alerts:unread_count")
            except Exception:
                pass
        return affected > 0

    def count_crisis_sessions(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM alerts"
            ).fetchone()[0]

    # ── RLHF queue operations ─────────────────────────────────────────────────

    def enqueue_rlhf(self, session_id: str, user_message: str, ai_response: str,
                     sub_agent: str, safety_level: str) -> str:
        rid = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO rlhf_queue
                   (id, session_id, user_message, ai_response, sub_agent,
                    safety_level, timestamp)
                   VALUES (?,?,?,?,?,?,?)""",
                (rid, session_id, user_message, ai_response,
                 sub_agent, safety_level, time.time()),
            )
        return rid

    def get_pending_rlhf(self, limit: int = 20) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM rlhf_queue WHERE scored=0 ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_pending_rlhf(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM rlhf_queue WHERE scored=0"
            ).fetchone()[0]

    def score_rlhf(self, response_id: str, score: int, note: Optional[str],
                   preferred: Optional[str], scored_by: str) -> Optional[Dict]:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            affected = conn.execute(
                """UPDATE rlhf_queue
                   SET scored=1, score=?, clinician_note=?,
                       preferred_response=?, scored_by=?, scored_at=?
                   WHERE id=?""",
                (score, note, preferred, scored_by, now, response_id),
            ).rowcount
            if affected == 0:
                return None
            row = conn.execute(
                "SELECT * FROM rlhf_queue WHERE id=?", (response_id,)
            ).fetchone()
        return dict(row) if row else None

    # ── PHQ (delegates to SQLiteBackend if available) ─────────────────────────

    def get_phq_history(self, session_id: str) -> List[Dict]:
        if self.db and hasattr(self.db, "get_phq_history"):
            return self.db.get_phq_history(session_id)
        # Fallback: read from phq_scores if table exists in this db
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT score, severity, timestamp FROM phq_scores WHERE session_id=? ORDER BY timestamp",
                    (session_id,)
                ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def healthcheck(self) -> Dict:
        result = {"sqlite": False, "redis": False}
        try:
            with self._conn() as conn:
                conn.execute("SELECT 1")
            result["sqlite"] = True
        except Exception as e:
            result["sqlite_error"] = str(e)

        if self._redis:
            try:
                self._redis.ping()
                result["redis"] = True
            except Exception as e:
                result["redis_error"] = str(e)

        return result


# ── Singleton store ───────────────────────────────────────────────────────────
_store: Optional[PersistentStore] = None
_alert_counter = 0  # kept for backwards compat; actual IDs come from DB

def get_store() -> PersistentStore:
    global _store
    if _store is None:
        _store = PersistentStore()
    return _store


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="MindBridge API",
    description="AI-powered mental health companion — Phase 8 deployment",
    version="8.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend ─────────────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib as _pl

_STATIC = _pl.Path(__file__).resolve().parent / "static"
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
    @app.get("/", include_in_schema=False)
    async def _root():
        return FileResponse(str(_STATIC / "index.html"))


# ── Lazy-load heavy components ────────────────────────────────────────────────
_agent = None
_user_db = None
_session_bridge = None

def get_agent():
    global _agent
    if _agent is None:
        from agents.agent_orchestrator import AgentOrchestrator
        from llm.inference_bridge import InferenceBridge
        llm = InferenceBridge(
            use_openai=True,
            max_new_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            fallback_to_stub=True,
        )
        # Use same DATA_DIR as auth — so session memory is shared
        _agent = AgentOrchestrator(
            model_path=None,
            rag_dir=config.RAG_DIR or None,
            memory_dir=str(DATA_DIR),
            region=config.REGION,
        )
        _agent.llm = llm
        logger.info(f"AgentOrchestrator ready — memory_dir={DATA_DIR}")
    return _agent

# ── Shared data directory (Windows + Linux safe, same for all components) ────
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_DB_PATH = str(DATA_DIR / "users.db")

def get_user_db():
    global _user_db
    if _user_db is None:
        from auth.user_db import UserDB
        # All components share ONE users.db to avoid split-brain logins
        _user_db = UserDB(db_path=USERS_DB_PATH)
        logger.info(f"UserDB → {USERS_DB_PATH}")
    return _user_db

def get_bridge():
    global _session_bridge
    if _session_bridge is None:
        from auth.Session_bridge import SessionBridge
        # SessionBridge uses the same users.db AND same memory_dir as the agent
        _session_bridge = SessionBridge(
            memory_dir=str(DATA_DIR),
            users_db_path=USERS_DB_PATH,
        )
        logger.info(f"SessionBridge → data_dir={DATA_DIR}")
    return _session_bridge

# ── JWT helpers ───────────────────────────────────────────────────────────────
def create_token(user_id: str, role: str = "patient") -> str:
    exp = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_H)
    return jwt.encode({"sub": user_id, "role": role, "exp": exp}, JWT_SECRET, algorithm=JWT_ALGO)

def decode_token(token: str) -> Dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_auth(authorization: Optional[str] = Header(None)) -> Dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return decode_token(authorization[7:])

def require_clinician(claims: Dict = Depends(require_auth)) -> Dict:
    if claims.get("role") not in ("clinician", "admin"):
        raise HTTPException(status_code=403, detail="Clinician access required")
    return claims

# ═══════════════════════════════════════════════════════════════════════════════
# Auth endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    phone: str
    password: str
    name: Optional[str] = None

class LoginRequest(BaseModel):
    phone: str
    password: str

@app.post("/auth/register", tags=["auth"])
async def register(body: RegisterRequest):
    db = get_user_db()  # shares same users.db as SessionBridge
    try:
        result = db.register(
            phone=body.phone,
            password=body.password,
            display_name=body.name or "User",
            role="user",
        )
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error", "فشل إنشاء الحساب"))
        token = create_token(result["user_id"], role="user")
        return {"user_id": result["user_id"], "token": token, "message": "Account created"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class ClinicianRegisterRequest(BaseModel):
    phone: str
    password: str
    name: Optional[str] = None
    admin_secret: str  # must match CLINICIAN_ADMIN_SECRET env var

@app.post("/auth/register/clinician", tags=["auth"])
async def register_clinician(body: ClinicianRegisterRequest):
    """Register a clinician account. Requires admin_secret env var."""
    admin_secret = os.environ.get("CLINICIAN_ADMIN_SECRET", "mindbridge-clinician-secret")
    if body.admin_secret != admin_secret:
        raise HTTPException(status_code=403, detail="رمز المسؤول غير صحيح")
    db = get_user_db()
    try:
        result = db.register(
            phone=body.phone,
            password=body.password,
            display_name=body.name or "Clinician",
            role="clinician",
        )
        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error", "فشل إنشاء الحساب"))
        token = create_token(result["user_id"], role="clinician")
        return {"user_id": result["user_id"], "token": token, "role": "clinician", "message": "Clinician account created"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login", tags=["auth"])
async def login(body: LoginRequest):
    bridge = get_bridge()
    try:
        result = bridge.login_and_start(phone=body.phone, password=body.password)
        if not result.get("ok"):
            raise HTTPException(status_code=401, detail=result.get("error", "فشل تسجيل الدخول"))

        # SessionBridge returns: {"ok": True, "token": ..., "session_id": ..., "user": {...}}
        user       = result["user"]
        user_id    = user["user_id"]
        session_id = result["session_id"]
        role       = user.get("role", "user")

        # Issue our own JWT (bridge token is internal)
        jwt_token = create_token(user_id, role=role)
        return {
            "token":      jwt_token,
            "user_id":    user_id,
            "session_id": session_id,
            "role":       role,
            "name":       user.get("display_name", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/auth/me", tags=["auth"])
async def me(claims: Dict = Depends(require_auth)):
    return {"user_id": claims["sub"], "role": claims.get("role", "patient")}

# ═══════════════════════════════════════════════════════════════════════════════
# Clinician ↔ Patient linking
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/clinician/invite", tags=["clinician"])
async def create_patient_invite(claims: Dict = Depends(require_clinician)):
    """
    Clinician generates a one-time invite code to share with a patient.
    The patient redeems it via POST /patient/accept-invite.
    """
    db = get_user_db()
    result = db.create_invite(clinician_id=claims["sub"])
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to create invite"))
    return {
        "invite_code": result["invite_code"],
        "message": "Share this code with your patient. It is single-use.",
        "expires": "Never (but only one patient can use it)",
    }

@app.post("/patient/accept-invite", tags=["patient"])
async def accept_patient_invite(body: dict, claims: Dict = Depends(require_auth)):
    """
    Patient redeems an invite code to link to a clinician.
    Body: { "invite_code": "MB-XXXXXXXX" }
    """
    if claims.get("role") not in ("user", "patient"):
        raise HTTPException(status_code=403, detail="Only patients can accept invites")
    code = (body.get("invite_code") or "").strip().upper()
    if not code:
        raise HTTPException(status_code=422, detail="invite_code is required")
    db = get_user_db()
    result = db.accept_invite(patient_id=claims["sub"], invite_code=code)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to accept invite"))
    return {"message": "Successfully linked to your clinician.", "clinician_id": result["clinician_id"]}

@app.get("/clinician/my-patients", tags=["clinician"])
async def my_patients(claims: Dict = Depends(require_clinician)):
    """
    Return all patients actively linked to this clinician,
    along with their bridge sessions and latest alerts.
    """
    db = get_user_db()
    patients = db.get_my_patients(clinician_id=claims["sub"])
    store = get_store()
    bridge = get_bridge()

    enriched = []
    for p in patients:
        pid = p["patient_id"]
        # Get all sessions for this patient
        sessions = []
        try:
            sessions = bridge.get_user_sessions(user_id=pid, limit=5)
        except Exception:
            pass
        # Get latest crisis alert for any of their sessions
        latest_alert = None
        try:
            for sess in sessions:
                alerts = store.get_alerts_for_session(sess["session_id"], limit=1)
                if alerts:
                    latest_alert = alerts[0]
                    break
        except Exception:
            pass

        enriched.append({
            "patient_id":    pid,
            "display_name":  p.get("display_name", "Patient"),
            "linked_at":     p.get("linked_at"),
            "member_since":  p.get("member_since"),
            "sessions":      sessions,
            "session_count": len(sessions),
            "latest_alert":  latest_alert,
            "has_crisis":    bool(latest_alert and latest_alert.get("safety_level") in ("hard_escalate", "watchdog_veto")),
        })

    return {"patients": enriched, "total": len(enriched)}

@app.get("/clinician/patient/{patient_id}/sessions", tags=["clinician"])
async def patient_sessions(patient_id: str, claims: Dict = Depends(require_clinician)):
    """
    Get all sessions for a specific patient — only if they are linked to this clinician.
    """
    db = get_user_db()
    if not db.is_my_patient(clinician_id=claims["sub"], patient_id=patient_id):
        raise HTTPException(status_code=403, detail="This patient is not linked to your account")
    bridge = get_bridge()
    try:
        sessions = bridge.get_user_sessions(user_id=patient_id, limit=20)
    except Exception:
        sessions = []
    return {"patient_id": patient_id, "sessions": sessions, "count": len(sessions)}

@app.get("/clinician/patient/{patient_id}/turns/{session_id}", tags=["clinician"])
async def patient_turns(
    patient_id: str,
    session_id: str,
    limit: int = 20,
    claims: Dict = Depends(require_clinician),
):
    """
    Get conversation turns for a specific patient session.
    Clinician must be linked to the patient.
    """
    db = get_user_db()
    if not db.is_my_patient(clinician_id=claims["sub"], patient_id=patient_id):
        raise HTTPException(status_code=403, detail="This patient is not linked to your account")
    bridge = get_bridge()
    turns = bridge.get_history(session_id=session_id, limit=limit)
    return {"patient_id": patient_id, "session_id": session_id, "turns": turns}

@app.delete("/clinician/patient/{patient_id}", tags=["clinician"])
async def remove_patient(patient_id: str, claims: Dict = Depends(require_clinician)):
    """Clinician removes a patient from their panel."""
    db = get_user_db()
    ok = db.revoke_link(clinician_id=claims["sub"], patient_id=patient_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Link not found")
    return {"message": "Patient removed from your panel", "patient_id": patient_id}

@app.get("/patient/my-clinician", tags=["patient"])
async def my_clinician(claims: Dict = Depends(require_auth)):
    """Patient checks who their linked clinician is."""
    db = get_user_db()
    clinician = db.get_my_clinician(patient_id=claims["sub"])
    if not clinician:
        return {"linked": False, "clinician": None}
    return {"linked": True, "clinician": clinician}

def get_user_db():
    """Get the UserDB instance."""
    from auth.user_db import UserDB
    return UserDB(db_path=str(DATA_DIR / "users.db"))


class ChatRequest(BaseModel):
    session_id: str
    message: str
    mood_signal: Optional[dict] = None  # Phase 8: from MoodMirror.jsx
    lang_hint: Optional[str] = None     # 'ar' | 'en' | None (auto-detect from message)

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sub_agent: str
    safety_level: str
    latency_ms: float
    crisis_detected: bool = False
    escalated: bool = False

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    claims: Dict = Depends(require_auth),
):
    agent = get_agent()
    try:
        # Pass mood_signal and lang_hint to agent if it accepts them
        kwargs = {}
        if body.mood_signal:
            kwargs["mood_signal"] = body.mood_signal
        if body.lang_hint and body.lang_hint in ("ar", "en"):
            kwargs["user_locale"] = body.lang_hint
        response = agent.respond(session_id=body.session_id, user_message=body.message, **kwargs)
    except TypeError:
        # Agent doesn't accept extra kwargs yet — graceful fallback
        response = agent.respond(session_id=body.session_id, user_message=body.message)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail="Agent unavailable")

    crisis_detected = response.safety_level in ("hard_escalate", "critical")
    escalated = False

    if crisis_detected:
        escalated = True
        background_tasks.add_task(
            fire_crisis_webhook,
            session_id=body.session_id,
            user_id=claims["sub"],
            safety_level=response.safety_level,
            message_snippet=body.message[:120],
        )

    background_tasks.add_task(
        queue_for_rlhf,
        session_id=body.session_id,
        user_message=body.message,
        ai_response=response.text,
        sub_agent=response.sub_agent,
        safety_level=response.safety_level,
    )

    # Persist turn to SessionBridge (links auth ↔ therapy memory)
    background_tasks.add_task(
        persist_turn_to_bridge,
        session_id=body.session_id,
        user_message=body.message,
        ai_response=response.text,
        sub_agent=response.sub_agent,
        safety_level=response.safety_level,
    )

    return ChatResponse(
        session_id=body.session_id,
        reply=response.text,
        sub_agent=response.sub_agent,
        safety_level=response.safety_level,
        latency_ms=response.latency_ms,
        crisis_detected=crisis_detected,
        escalated=escalated,
    )

@app.get("/chat/history/{session_id}", tags=["chat"])
async def chat_history(session_id: str, claims: Dict = Depends(require_auth)):
    bridge = get_bridge()
    try:
        history = bridge.get_session_history(session_id)
        return {"session_id": session_id, "turns": history}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# Crisis Webhook — now persisted to DB
# ═══════════════════════════════════════════════════════════════════════════════

async def fire_crisis_webhook(session_id: str, user_id: str, safety_level: str, message_snippet: str):
    store = get_store()
    escalated_at = datetime.utcnow().isoformat()

    alert_id = store.add_alert(
        session_id=session_id,
        user_id=user_id,
        safety_level=safety_level,
        message_snippet=message_snippet,
        escalated_at=escalated_at,
    )
    logger.warning(f"🚨 CRISIS ALERT #{alert_id} | session={session_id} | level={safety_level} [persisted]")

    payload = {
        "event": "mindbridge_crisis",
        "alert_id": alert_id,
        "session_id": session_id,
        "safety_level": safety_level,
        "message": "Crisis detected — patient needs immediate attention",
        "timestamp": escalated_at,
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        if CRISIS_WEBHOOK_URL:
            try:
                await client.post(CRISIS_WEBHOOK_URL, json=payload)
            except Exception as e:
                logger.error(f"Crisis webhook failed: {e}")
        if SMS_WEBHOOK_URL:
            try:
                await client.post(SMS_WEBHOOK_URL, json={**payload, "channel": "sms"})
            except Exception as e:
                logger.error(f"SMS webhook failed: {e}")
        if PUSH_WEBHOOK_URL:
            try:
                await client.post(PUSH_WEBHOOK_URL, json={**payload, "channel": "push"})
            except Exception as e:
                logger.error(f"Push webhook failed: {e}")

@app.post("/crisis/webhook", tags=["crisis"], include_in_schema=False)
async def internal_crisis_webhook(request: Request):
    body = await request.json()
    await fire_crisis_webhook(
        session_id=body.get("session_id", "unknown"),
        user_id=body.get("user_id", "unknown"),
        safety_level=body.get("safety_level", "hard_escalate"),
        message_snippet=body.get("message", "")[:120],
    )
    return {"status": "received"}

class CrisisTestRequest(BaseModel):
    session_id: str = "test-session-001"
    message: str = "اريد ان انهي حياتي"

@app.post("/crisis/test", tags=["crisis"])
async def test_crisis(body: CrisisTestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        fire_crisis_webhook,
        session_id=body.session_id,
        user_id="test-user",
        safety_level="hard_escalate",
        message_snippet=body.message,
    )
    return {"status": "crisis_simulated", "session_id": body.session_id}

# ═══════════════════════════════════════════════════════════════════════════════
# Clinician Dashboard — persistent alerts
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/dashboard/alerts", tags=["dashboard"])
async def dashboard_alerts(
    unread_only: bool = False,
    limit: int = 50,
    claims: Dict = Depends(require_clinician),
):
    store = get_store()
    alerts = store.get_alerts(unread_only=unread_only, limit=limit)
    unread = store.count_unread_alerts()
    return {
        "alerts": alerts,
        "total": len(alerts),
        "unread": unread,
    }

@app.post("/dashboard/alerts/{alert_id}/ack", tags=["dashboard"])
async def ack_alert(alert_id: int, claims: Dict = Depends(require_clinician)):
    store = get_store()
    ok = store.ack_alert(alert_id=alert_id, acknowledged_by=claims["sub"])
    if not ok:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged", "alert_id": alert_id}

@app.get("/dashboard/sessions", tags=["dashboard"])
async def dashboard_sessions(claims: Dict = Depends(require_clinician)):
    """Return sessions only for patients linked to this clinician."""
    db = get_user_db()
    bridge = get_bridge()
    store = get_store()

    patients = db.get_my_patients(clinician_id=claims["sub"])
    sessions = []
    for p in patients:
        try:
            pat_sessions = bridge.get_user_sessions(user_id=p["patient_id"], limit=10)
            for s in pat_sessions:
                s["patient_id"]   = p["patient_id"]
                s["patient_name"] = p.get("display_name", "Patient")
                # Attach latest risk level from alerts
                try:
                    alerts = store.get_alerts_for_session(s["session_id"], limit=1)
                    s["risk_trend"] = alerts[0]["safety_level"] if alerts else "stable"
                except Exception:
                    s["risk_trend"] = "stable"
            sessions.extend(pat_sessions)
        except Exception:
            pass

    sessions.sort(key=lambda s: s.get("last_active", 0), reverse=True)
    return {"sessions": sessions, "count": len(sessions)}

@app.get("/dashboard/stats", tags=["dashboard"])
async def dashboard_stats(claims: Dict = Depends(require_clinician)):
    store = get_store()
    return {
        "total_alerts": len(store.get_alerts(limit=99999)),
        "unread_alerts": store.count_unread_alerts(),
        "crisis_sessions": store.count_crisis_sessions(),
        "rlhf_pending": store.count_pending_rlhf(),
        "timestamp": datetime.utcnow().isoformat(),
    }

# ── NEW Phase 8: PHQ endpoint for Clinician.jsx tab 3 ─────────────────────────
@app.get("/dashboard/phq/{session_id}", tags=["dashboard"])
async def dashboard_phq(
    session_id: str,
    claims: Dict = Depends(require_clinician),
):
    """Return PHQ-8 history for a session — used by the PHQ chart in Clinician.jsx."""
    store = get_store()
    phq_history = store.get_phq_history(session_id)
    latest = phq_history[-1] if phq_history else None
    trend = None
    if len(phq_history) >= 2:
        if phq_history[-1]["score"] < phq_history[-2]["score"]:
            trend = "improving"
        elif phq_history[-1]["score"] > phq_history[-2]["score"]:
            trend = "worsening"
        else:
            trend = "stable"
    return {
        "session_id": session_id,
        "phq_history": phq_history,
        "latest": latest,
        "trend": trend,
        "count": len(phq_history),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# RLHF — persistent queue
# ═══════════════════════════════════════════════════════════════════════════════

async def persist_turn_to_bridge(session_id: str, user_message: str, ai_response: str,
                                  sub_agent: str, safety_level: str):
    """Write each chat turn into SessionBridge so auth + therapy memory stay in sync."""
    try:
        bridge = get_bridge()
        bridge.add_turn(
            session_id=session_id,
            user=user_message,
            assistant=ai_response,
            metadata={"sub_agent": sub_agent, "safety_level": safety_level},
        )
    except Exception as e:
        logger.warning(f"persist_turn_to_bridge failed (non-fatal): {e}")

async def queue_for_rlhf(session_id: str, user_message: str, ai_response: str,
                          sub_agent: str, safety_level: str):
    store = get_store()
    store.enqueue_rlhf(
        session_id=session_id,
        user_message=user_message,
        ai_response=ai_response,
        sub_agent=sub_agent,
        safety_level=safety_level,
    )

@app.get("/rlhf/pending", tags=["rlhf"])
async def rlhf_pending(limit: int = 20, claims: Dict = Depends(require_clinician)):
    store = get_store()
    pending = store.get_pending_rlhf(limit=limit)
    return {"pending": pending, "total_pending": store.count_pending_rlhf()}

class ScoreRequest(BaseModel):
    response_id: str
    score: int = Field(..., ge=1, le=5, description="1=poor … 5=excellent")
    preferred_response: Optional[str] = None
    note: Optional[str] = None

@app.post("/rlhf/score", tags=["rlhf"])
async def rlhf_score(body: ScoreRequest, claims: Dict = Depends(require_clinician)):
    store = get_store()
    item = store.score_rlhf(
        response_id=body.response_id,
        score=body.score,
        note=body.note,
        preferred=body.preferred_response,
        scored_by=claims["sub"],
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Response not found")

    # Write DPO pair to disk if preferred_response given and score is low
    if body.preferred_response and body.score <= 2:
        pair = {
            "prompt": item["user_message"],
            "chosen": body.preferred_response,
            "rejected": item["ai_response"],
            "session_id": item["session_id"],
            "scored_by": claims["sub"],
        }
        dpo_path = ROOT / "rlhf" / "dpo_pairs.jsonl"
        try:
            with open(dpo_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Could not write DPO pair: {e}")

    return {"status": "scored", "response_id": body.response_id, "score": body.score}

# ═══════════════════════════════════════════════════════════════════════════════
# Health & status
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["system"])
async def health():
    store = get_store()
    storage_health = store.healthcheck()
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "storage": storage_health,
    }

@app.get("/status", tags=["system"])
async def status():
    store = get_store()
    return {
        "service": "MindBridge API",
        "phase": 8,
        "version": "8.0.0",
        "config": config.summary(),
        "alerts_total": len(store.get_alerts(limit=99999)),
        "rlhf_pending": store.count_pending_rlhf(),
        "storage": store.healthcheck(),
        "webhooks": {
            "crisis": bool(CRISIS_WEBHOOK_URL),
            "sms": bool(SMS_WEBHOOK_URL),
            "push": bool(PUSH_WEBHOOK_URL),
        },
    }




# ═══ /api/* compat routes ═══════════════════════════════════════════════════
@app.post("/api/session/new", tags=["compat"])
async def _api_session_new():
    import uuid
    return {"session_id": "web-" + uuid.uuid4().hex[:16]}

@app.get("/api/session/{session_id}/progress", tags=["compat"])
async def _api_progress(session_id: str, claims: Dict = Depends(require_auth)):
    try:
        agent = get_agent()
        rec   = agent.memory.get_session(session_id)
        if not rec:
            return {"session_id":session_id,"total_turns":0,"mood_history":[],"active_goals":[],"summary":None}
        return agent.memory.get_progress_report(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mood", tags=["compat"])
async def _api_mood(request: Request, claims: Dict = Depends(require_auth)):
    data = await request.json()
    sid  = data.get("session_id")
    if not sid: raise HTTPException(status_code=400, detail="session_id required")
    try:
        agent = get_agent()
        agent.memory.log_mood(sid, int(data.get("mood",5)), int(data.get("energy",5)), int(data.get("sleep",5)))
        return {"ok":True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ritual", tags=["compat"])
async def _api_ritual(request: Request, claims: Dict = Depends(require_auth)):
    data = await request.json()
    sid  = data.get("session_id","web-default")
    score= int(data.get("mood_score",5))
    ctx  = data.get("context","session")
    try:
        agent = get_agent()
        from agents.features.ritual_planner import RitualPlannerAgent
        planner = RitualPlannerAgent(llm=agent.llm)
        rec     = planner.recommend(sid, score, ctx)
        return {"ritual_type":rec.ritual_type,"ritual_name":rec.ritual_name,
                "duration_min":rec.duration_min,"why":rec.why,
                "steps":rec.steps,"after_prompt":rec.after_prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", tags=["compat"])
async def _api_status():
    store = get_store()
    return {"service":"MindBridge","model": getattr(config,"GROQ_MODEL","llama-3.3-70b-versatile"),
            "storage":store.healthcheck()}

@app.get("/api/health", tags=["compat"])
async def _api_health():
    from datetime import datetime
    return {"status":"ok","timestamp":datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
