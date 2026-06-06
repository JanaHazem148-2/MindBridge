"""
auth/Session_bridge.py
────────────────────────
Bridges user authentication (UserDB) with session management
(SQLite-backed memory store).

Provides:
  SessionBridge.login_and_start(phone, password)  → creates/resumes session
  SessionBridge.add_turn(session_id, user, assistant, metadata)
  SessionBridge.get_history(session_id)
  SessionBridge.get_user_sessions(user_id)
"""

import uuid
import time
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from contextlib import contextmanager

from auth.user_db import UserDB

logger = logging.getLogger(__name__)


SESSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS bridge_sessions (
    session_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    created_at    REAL NOT NULL,
    last_active   REAL NOT NULL,
    total_turns   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_bs_user ON bridge_sessions(user_id);

CREATE TABLE IF NOT EXISTS bridge_turns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    user_msg      TEXT,
    assistant_msg TEXT,
    sub_agent     TEXT,
    safety_level  TEXT,
    timestamp     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bt_session ON bridge_turns(session_id);
"""


class SessionBridge:
    """
    Glue layer between user accounts and conversation sessions.
    """

    def __init__(
        self,
        memory_dir: str = "./data",
        users_db_path: Optional[str] = None,
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        db_path = users_db_path or str(self.memory_dir / "users.db")
        self.user_db = UserDB(db_path=db_path)

        self._sessions_db = str(self.memory_dir / "bridge_sessions.db")
        self._ensure_schema()
        logger.info(f"SessionBridge → {self._sessions_db}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._sessions_db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self):
        with self._conn() as conn:
            conn.executescript(SESSIONS_SCHEMA)

    # ── Auth + session ─────────────────────────────────────────────────────────

    def login_and_start(self, phone: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user and return an active session_id.
        Creates a new session each login (sessions are per-conversation).
        """
        result = self.user_db.login(phone=phone, password=password)
        if not result.get("ok"):
            return result

        user       = result["user"]
        session_id = "web-" + uuid.uuid4().hex[:16]
        now        = time.time()

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO bridge_sessions (session_id, user_id, created_at, last_active)
                   VALUES (?,?,?,?)""",
                (session_id, user["user_id"], now, now),
            )

        logger.info(f"SessionBridge: login ok → session {session_id}")
        return {
            "ok":        True,
            "session_id": session_id,
            "user":      user,
            # token is issued by api.py (JWT), bridge doesn't manage JWTs
        }

    def add_turn(
        self,
        session_id: str,
        user: str,
        assistant: str,
        metadata: Optional[Dict] = None,
    ):
        """Persist a conversation turn."""
        meta = metadata or {}
        now  = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO bridge_turns
                   (session_id, user_msg, assistant_msg, sub_agent, safety_level, timestamp)
                   VALUES (?,?,?,?,?,?)""",
                (
                    session_id, user, assistant,
                    meta.get("sub_agent", ""),
                    meta.get("safety_level", "safe"),
                    now,
                ),
            )
            conn.execute(
                """UPDATE bridge_sessions
                   SET last_active=?, total_turns=total_turns+1
                   WHERE session_id=?""",
                (now, session_id),
            )

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Return last `limit` turns for a session."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT user_msg, assistant_msg, sub_agent, safety_level, timestamp
                   FROM bridge_turns WHERE session_id=?
                   ORDER BY timestamp DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict]:
        """List sessions for a user."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT session_id, created_at, last_active, total_turns
                   FROM bridge_sessions WHERE user_id=?
                   ORDER BY last_active DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
