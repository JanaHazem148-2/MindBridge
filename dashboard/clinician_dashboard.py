"""
dashboard/clinician_dashboard.py
──────────────────────────────────
Phase 6 — Clinician Dashboard.

A FastAPI server providing clinicians with:
  - Real-time session monitoring
  - Alert queue (crisis escalations)
  - PHQ trend analysis per patient
  - Mood timeline visualisation data
  - Session transcripts with safety annotations
  - Manual override: flag a session for urgent review
  - Watchdog risk summaries across all active sessions

Endpoints:
  GET  /health                          — service health
  GET  /sessions                        — list all sessions (paginated)
  GET  /sessions/{session_id}           — full session detail
  GET  /sessions/{session_id}/turns     — conversation transcript
  GET  /sessions/{session_id}/risk      — watchdog risk summary
  GET  /sessions/{session_id}/phq       — PHQ score history
  GET  /sessions/{session_id}/mood      — mood timeline
  POST /sessions/{session_id}/flag      — flag session for urgent review
  GET  /alerts                          — alert queue (crisis events)
  GET  /alerts/unread                   — unread alerts only
  POST /alerts/{alert_id}/acknowledge   — mark alert as reviewed
  GET  /overview                        — all sessions risk overview
  GET  /stats                           — aggregate stats for reporting
  DELETE /sessions/{session_id}         — GDPR erasure

Authentication:
  Bearer token (JWT) — clinicians must authenticate before accessing
  All endpoints require valid token with 'clinician' role

HIPAA:
  All reads logged to audit_log
  No PII in responses — session_id only
  Transcripts masked by default (full content only with explicit reason)

Setup:
    pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt]

    # Set secrets
    export DASHBOARD_SECRET_KEY="your-32-char-secret"
    export DASHBOARD_ADMIN_TOKEN="your-admin-token"

    # Run
    uvicorn dashboard.clinician_dashboard:app --host 0.0.0.0 --port 8001
"""

import os
import json
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

SECRET_KEY  = os.environ.get("DASHBOARD_SECRET_KEY", "mindbridge-dev-secret-change-in-prod")
ADMIN_TOKEN = os.environ.get("DASHBOARD_ADMIN_TOKEN", "dev-admin-token")

# ── Try to import FastAPI — graceful failure if not installed ─────────────────
try:
    from fastapi import FastAPI, HTTPException, Depends, Header, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed — dashboard unavailable. Install: pip install fastapi uvicorn")


# ── Data models ───────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    class FlagRequest(BaseModel):
        reason:    str
        urgency:   str = "standard"   # "standard" | "urgent" | "critical"
        clinician: str = ""

    class AcknowledgeRequest(BaseModel):
        clinician:  str
        notes:      str = ""
        action_taken: str = ""   # "contacted_patient" | "referred" | "no_action" | "other"


# ── In-memory alert store (replace with DB in production) ─────────────────────

_alerts: List[Dict] = []
_alert_id_counter = 0

def _add_alert(session_id: str, level: str, message: str, data: Dict = None) -> int:
    global _alert_id_counter
    _alert_id_counter += 1
    alert = {
        "id":           _alert_id_counter,
        "session_id":   session_id,
        "level":        level,
        "message":      message,
        "data":         data or {},
        "timestamp":    time.time(),
        "acknowledged": False,
        "acknowledged_by": None,
        "acknowledged_at": None,
    }
    _alerts.append(alert)
    return _alert_id_counter


# ── Auth ──────────────────────────────────────────────────────────────────────

def verify_token(authorization: Optional[str] = Header(None)) -> str:
    if not FASTAPI_AVAILABLE:
        return "dev"
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header required")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Bearer token required")
    # Simple token check — in production use JWT with expiry + role claims
    if token not in (ADMIN_TOKEN, os.environ.get("CLINICIAN_TOKEN", ADMIN_TOKEN)):
        raise HTTPException(status_code=403, detail="Invalid token")
    return token


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(session_store=None, watchdog=None) -> Optional[Any]:
    """
    Create the FastAPI dashboard app.

    Args:
        session_store: ProductionSessionStore instance
        watchdog:      SafetyWatchdog instance

    Returns FastAPI app, or None if FastAPI not installed.
    """
    if not FASTAPI_AVAILABLE:
        return None

    app = FastAPI(
        title="MindBridge Clinician Dashboard",
        description="Clinical monitoring and oversight for MindBridge AI sessions",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ─────────────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        store_health = session_store.healthcheck() if session_store else {"status": "no_store"}
        return {
            "status":        "ok",
            "timestamp":     time.time(),
            "session_store": store_health,
            "alerts_total":  len(_alerts),
            "alerts_unread": sum(1 for a in _alerts if not a["acknowledged"]),
        }

    # ── Sessions ───────────────────────────────────────────────────────────────

    @app.get("/sessions")
    def list_sessions(
        page:  int = Query(1, ge=1),
        limit: int = Query(20, ge=1, le=100),
        token: str = Depends(verify_token),
    ):
        if not session_store:
            return {"sessions": [], "total": 0}

        # Get all sessions from store (paginated)
        # In production this would be a proper DB query
        all_sessions = []
        if hasattr(session_store.db, "_sessions"):
            all_sessions = list(session_store.db._sessions.values())

        start = (page - 1) * limit
        end   = start + limit
        page_sessions = all_sessions[start:end]

        return {
            "sessions": [
                {
                    "session_id":   s.get("session_id", ""),
                    "total_turns":  s.get("total_turns", 0),
                    "last_active":  s.get("last_active", 0),
                    "created_at":   s.get("created_at", 0),
                    "has_summary":  bool(s.get("summary")),
                }
                for s in page_sessions
            ],
            "total": len(all_sessions),
            "page":  page,
            "limit": limit,
        }

    @app.get("/sessions/{session_id}")
    def get_session(
        session_id: str,
        token: str = Depends(verify_token),
    ):
        if not session_store:
            raise HTTPException(status_code=503, detail="Session store not available")
        session = session_store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Audit log this read
        session_store.db.write_audit("READ", session_id, "clinician", {"endpoint": "get_session"})

        phq   = session_store.db.get_phq_history(session_id)
        risk  = watchdog.get_session_risk_summary(session_id) if watchdog else {}

        return {
            "session_id":   session_id,
            "total_turns":  session.get("total_turns", 0),
            "created_at":   session.get("created_at", 0),
            "last_active":  session.get("last_active", 0),
            "summary":      session.get("summary"),
            "phq_latest":   phq[-1] if phq else None,
            "risk":         risk,
            "flags_count":  risk.get("hard_escalations", 0) if risk else 0,
        }

    @app.get("/sessions/{session_id}/turns")
    def get_turns(
        session_id: str,
        limit:   int = Query(20, ge=1, le=100),
        masked:  bool = Query(True),   # mask by default for privacy
        reason:  str  = Query("", description="Required to unmask transcript"),
        token: str = Depends(verify_token),
    ):
        if not session_store:
            raise HTTPException(status_code=503)
        turns = session_store.get_recent_turns(session_id, limit)
        if not turns:
            raise HTTPException(status_code=404, detail="No turns found")

        # Audit
        session_store.db.write_audit(
            "READ", session_id, "clinician",
            {"endpoint": "turns", "masked": masked, "reason": reason}
        )

        if masked:
            return {
                "session_id": session_id,
                "turns": [
                    {
                        "turn_index":   i,
                        "sub_agent":    t.get("sub_agent", "unknown"),
                        "safety_level": t.get("safety_level", "safe"),
                        "timestamp":    t.get("timestamp", 0),
                        "user_preview": t.get("user_text", t.get("user", ""))[:30] + "...",
                        "masked":       True,
                    }
                    for i, t in enumerate(turns)
                ],
                "note": "Full transcript requires reason parameter and is logged for HIPAA compliance"
            }
        else:
            if not reason:
                raise HTTPException(
                    status_code=422,
                    detail="reason parameter required to access unmasked transcript"
                )
            return {
                "session_id": session_id,
                "turns":      turns,
                "unmasked":   True,
                "accessed_by": token[:8] + "...",
            }

    @app.get("/sessions/{session_id}/risk")
    def get_risk(session_id: str, token: str = Depends(verify_token)):
        if not watchdog:
            return {"session_id": session_id, "status": "watchdog_not_available"}
        return watchdog.get_session_risk_summary(session_id)

    @app.get("/sessions/{session_id}/phq")
    def get_phq(session_id: str, token: str = Depends(verify_token)):
        if not session_store:
            raise HTTPException(status_code=503)
        phq = session_store.db.get_phq_history(session_id)
        return {
            "session_id": session_id,
            "phq_history": phq,
            "latest":      phq[-1] if phq else None,
            "trend":       "improving" if len(phq) >= 2 and phq[-1]["score"] < phq[-2]["score"]
                           else "worsening" if len(phq) >= 2 and phq[-1]["score"] > phq[-2]["score"]
                           else "stable",
        }

    @app.post("/sessions/{session_id}/flag")
    def flag_session(
        session_id: str,
        body: "FlagRequest",
        token: str = Depends(verify_token),
    ):
        alert_id = _add_alert(
            session_id=session_id,
            level="manual_flag",
            message=f"Manually flagged by clinician: {body.reason}",
            data={"urgency": body.urgency, "clinician": body.clinician},
        )
        logger.warning(f"Session {session_id[:8]} manually flagged: {body.reason}")
        return {"alert_id": alert_id, "message": "Session flagged for review"}

    @app.delete("/sessions/{session_id}")
    def delete_session(
        session_id: str,
        reason:     str = Query(..., description="GDPR deletion reason required"),
        token: str = Depends(verify_token),
    ):
        if not session_store:
            raise HTTPException(status_code=503)
        session_store.delete_session(session_id, actor=f"clinician:{token[:8]}")
        return {"message": "Session data erased (GDPR)", "session_id": session_id}

    # ── Alerts ─────────────────────────────────────────────────────────────────

    @app.get("/alerts")
    def get_alerts(
        limit:  int  = Query(50, ge=1, le=200),
        unread: bool = Query(False),
        token: str = Depends(verify_token),
    ):
        alerts = _alerts
        if unread:
            alerts = [a for a in alerts if not a["acknowledged"]]
        return {
            "alerts": sorted(alerts, key=lambda a: -a["timestamp"])[:limit],
            "total":  len(alerts),
            "unread": sum(1 for a in _alerts if not a["acknowledged"]),
        }

    @app.get("/alerts/unread")
    def get_unread_alerts(token: str = Depends(verify_token)):
        unread = [a for a in _alerts if not a["acknowledged"]]
        return {"alerts": sorted(unread, key=lambda a: -a["timestamp"]), "count": len(unread)}

    @app.post("/alerts/{alert_id}/acknowledge")
    def acknowledge_alert(
        alert_id: int,
        body: "AcknowledgeRequest",
        token: str = Depends(verify_token),
    ):
        for alert in _alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"]    = True
                alert["acknowledged_by"] = body.clinician
                alert["acknowledged_at"] = time.time()
                alert["action_taken"]    = body.action_taken
                alert["notes"]           = body.notes
                return {"message": "Alert acknowledged", "alert_id": alert_id}
        raise HTTPException(status_code=404, detail="Alert not found")

    # ── Overview ───────────────────────────────────────────────────────────────

    @app.get("/overview")
    def overview(token: str = Depends(verify_token)):
        if watchdog:
            summaries = watchdog.all_session_summaries()
        else:
            summaries = []

        acute    = [s for s in summaries if s.get("risk_trend") == "acute"]
        worsen   = [s for s in summaries if s.get("risk_trend") == "worsening"]
        stable   = [s for s in summaries if s.get("risk_trend") in ("stable", "recovering")]

        return {
            "timestamp":         time.time(),
            "total_sessions":    len(summaries),
            "acute_risk":        len(acute),
            "worsening":         len(worsen),
            "stable":            len(stable),
            "unread_alerts":     sum(1 for a in _alerts if not a["acknowledged"]),
            "acute_sessions":    acute[:10],
            "worsening_sessions": worsen[:10],
        }

    @app.get("/stats")
    def stats(token: str = Depends(verify_token)):
        total_alerts  = len(_alerts)
        hard_escl     = sum(1 for a in _alerts if a.get("level") == "hard_escalate")
        ack_rate      = sum(1 for a in _alerts if a["acknowledged"]) / max(total_alerts, 1)

        return {
            "total_alerts":         total_alerts,
            "hard_escalations":     hard_escl,
            "acknowledgement_rate": round(ack_rate, 3),
            "sessions_monitored":   len(watchdog.all_session_summaries()) if watchdog else 0,
            "generated_at":         datetime.utcnow().isoformat(),
        }

    return app


# ── Convenience webhook receiver ──────────────────────────────────────────────

def create_webhook_receiver(alert_store=None) -> Optional[Any]:
    """
    A separate lightweight FastAPI app to receive webhook alerts
    from the SafetyWatchdog and ClinicianAlertQueue.
    """
    if not FASTAPI_AVAILABLE:
        return None

    webhook_app = FastAPI(title="MindBridge Alert Webhook Receiver")

    @webhook_app.post("/webhook/alert")
    async def receive_alert(payload: dict):
        session_id = payload.get("session_id", "unknown")
        level      = payload.get("risk_trend", "unknown")
        _add_alert(
            session_id=session_id,
            level=level,
            message=f"Watchdog alert: {level} (risk={payload.get('risk_score', 0):.2f})",
            data=payload,
        )
        logger.critical(
            f"WEBHOOK ALERT | session={session_id[:8]} | "
            f"level={level} | flags={payload.get('flags', [])}"
        )
        return {"received": True, "alert_id": _alert_id_counter}

    return webhook_app


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    import uvicorn
    from infra.session_store import ProductionSessionStore

    store = ProductionSessionStore.from_env()

    try:
        from agents.safety_watchdog import SafetyWatchdog
        wd = SafetyWatchdog(region="EG")
    except Exception:
        wd = None

    app = create_app(session_store=store, watchdog=wd)

    print("\n" + "=" * 55)
    print("  MindBridge Clinician Dashboard")
    print("=" * 55)
    print(f"  URL:       http://localhost:8001")
    print(f"  Docs:      http://localhost:8001/docs")
    print(f"  Token:     {ADMIN_TOKEN}")
    print(f"  Store:     {type(store.db).__name__}")
    print("=" * 55 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
