"""
agents/session_memory.py
─────────────────────────
Persistent session memory for the MindBridge agent.

Stores per-session:
  - Conversation turns (user/assistant pairs)
  - PHQ scores + mood logs over time
  - Active therapeutic goals
  - Safety flags and escalation history
  - Session summary (auto-updated every N turns)

Storage backends:
  - SQLite  (default — works on any server, no extra infra)
  - In-memory dict (testing / ephemeral deployments)

Schema (SQLite):
  sessions(session_id, created_at, last_active, total_turns, summary, metadata_json)
  turns(id, session_id, turn_index, user, assistant, timestamp, metadata_json)
  phq_scores(id, session_id, score, severity, timestamp)
  mood_logs(id, session_id, mood, energy, sleep, timestamp)
  goals(id, session_id, goal_text, status, created_at, updated_at)
  safety_events(id, session_id, level, triggered_by, matched_json, timestamp)
"""

import os
import json
import time
import sqlite3
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    user:       str
    assistant:  str
    timestamp:  float = field(default_factory=time.time)
    metadata:   Dict  = field(default_factory=dict)


@dataclass
class PHQRecord:
    score:     int
    severity:  str
    timestamp: float = field(default_factory=time.time)


@dataclass
class MoodRecord:
    mood:      int    # 1-10
    energy:    int    # 1-10
    sleep:     int    # 1-10
    timestamp: float = field(default_factory=time.time)


@dataclass
class GoalRecord:
    goal_id:    str
    goal_text:  str
    status:     str   # "active" | "completed" | "paused"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class SessionRecord:
    session_id:       str
    created_at:       float
    last_active:      float
    total_turns:      int
    summary:          Optional[str]
    recent_turns:     List[Dict]       # last N turns as dicts
    phq_history:      List[PHQRecord]
    mood_history:     List[MoodRecord]
    goals:            List[GoalRecord]
    active_assessment: bool = False    # True if mid-PHQ-8
    metadata:         Dict = field(default_factory=dict)


# ── Memory backend ────────────────────────────────────────────────────────────

class SessionMemory:
    """
    Thread-safe session memory with SQLite persistence.

    Hot cache: keeps recently active sessions in RAM for fast reads.
    Persistence: writes every turn to SQLite so nothing is lost on restart.
    """

    HOT_CACHE_SIZE = 500   # sessions kept in memory

    def __init__(self, persist_dir: Optional[str] = None, backend: str = "sqlite"):
        self.persist_dir = persist_dir or "/tmp/mindbridge_memory"
        self.backend     = backend
        self._lock       = threading.Lock()
        self._hot_cache: Dict[str, SessionRecord] = {}

        if backend == "sqlite":
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._db_path = os.path.join(self.persist_dir, "sessions.db")
            self._init_db()
        else:
            self._db_path = None

        logger.info(f"SessionMemory: backend={backend} | dir={self.persist_dir}")

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    created_at   REAL,
                    last_active  REAL,
                    total_turns  INTEGER DEFAULT 0,
                    summary      TEXT,
                    active_assessment INTEGER DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT,
                    turn_index   INTEGER,
                    user         TEXT,
                    assistant    TEXT,
                    timestamp    REAL,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS phq_scores (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT,
                    score        INTEGER,
                    severity     TEXT,
                    timestamp    REAL
                );
                CREATE TABLE IF NOT EXISTS mood_logs (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT,
                    mood         INTEGER,
                    energy       INTEGER,
                    sleep        INTEGER,
                    timestamp    REAL
                );
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id      TEXT PRIMARY KEY,
                    session_id   TEXT,
                    goal_text    TEXT,
                    status       TEXT DEFAULT 'active',
                    created_at   REAL,
                    updated_at   REAL
                );
                CREATE TABLE IF NOT EXISTS safety_events (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT,
                    level        TEXT,
                    triggered_by TEXT,
                    matched_json TEXT,
                    timestamp    REAL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
                CREATE INDEX IF NOT EXISTS idx_phq_session   ON phq_scores(session_id);
                CREATE INDEX IF NOT EXISTS idx_mood_session  ON mood_logs(session_id);
                CREATE INDEX IF NOT EXISTS idx_goals_session ON goals(session_id);
            """)

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Return SessionRecord or None if session doesn't exist yet."""
        with self._lock:
            if session_id in self._hot_cache:
                return self._hot_cache[session_id]
        return self._load_from_db(session_id)

    def _load_from_db(self, session_id: str) -> Optional[SessionRecord]:
        if self.backend != "sqlite":
            return None
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if not row:
                return None

            # Load last 20 turns
            turns = conn.execute(
                "SELECT user, assistant, timestamp, metadata_json FROM turns "
                "WHERE session_id=? ORDER BY turn_index DESC LIMIT 20",
                (session_id,)
            ).fetchall()
            recent_turns = [
                {"user": t["user"], "assistant": t["assistant"],
                 "timestamp": t["timestamp"],
                 **json.loads(t["metadata_json"] or "{}")}
                for t in reversed(turns)
            ]

            # PHQ history
            phq_rows = conn.execute(
                "SELECT score, severity, timestamp FROM phq_scores WHERE session_id=? ORDER BY timestamp",
                (session_id,)
            ).fetchall()
            phq_history = [PHQRecord(r["score"], r["severity"], r["timestamp"]) for r in phq_rows]

            # Mood history
            mood_rows = conn.execute(
                "SELECT mood, energy, sleep, timestamp FROM mood_logs WHERE session_id=? ORDER BY timestamp",
                (session_id,)
            ).fetchall()
            mood_history = [MoodRecord(r["mood"], r["energy"], r["sleep"], r["timestamp"]) for r in mood_rows]

            # Goals
            goal_rows = conn.execute(
                "SELECT goal_id, goal_text, status, created_at, updated_at FROM goals WHERE session_id=?",
                (session_id,)
            ).fetchall()
            goals = [GoalRecord(g["goal_id"], g["goal_text"], g["status"],
                                g["created_at"], g["updated_at"]) for g in goal_rows]

        record = SessionRecord(
            session_id=session_id,
            created_at=row["created_at"],
            last_active=row["last_active"],
            total_turns=row["total_turns"],
            summary=row["summary"],
            recent_turns=recent_turns,
            phq_history=phq_history,
            mood_history=mood_history,
            goals=goals,
            active_assessment=bool(row["active_assessment"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
        with self._lock:
            self._hot_cache[session_id] = record
        return record

    def append_turn(
        self,
        session_id: str,
        user: str,
        assistant: str,
        metadata: Optional[Dict] = None,
    ) -> List[str]:
        """
        Append a conversation turn. Creates the session if it doesn't exist.
        Returns list of memory keys updated.
        """
        now = time.time()
        metadata = metadata or {}
        updated_keys = [f"turns:{session_id}"]

        record = self.get_session(session_id)
        if record is None:
            record = SessionRecord(
                session_id=session_id,
                created_at=now,
                last_active=now,
                total_turns=0,
                summary=None,
                recent_turns=[],
                phq_history=[],
                mood_history=[],
                goals=[],
            )

        turn_index = record.total_turns
        record.total_turns += 1
        record.last_active  = now
        record.recent_turns.append({
            "user": user, "assistant": assistant,
            "timestamp": now, **metadata
        })
        # Keep hot cache lean
        if len(record.recent_turns) > 30:
            record.recent_turns = record.recent_turns[-30:]

        # Log safety events
        safety_level = metadata.get("safety_level", "safe")
        if safety_level not in ("safe",):
            updated_keys.append(f"safety:{session_id}")

        if self.backend == "sqlite":
            with sqlite3.connect(self._db_path) as conn:
                # Upsert session
                conn.execute("""
                    INSERT INTO sessions(session_id, created_at, last_active, total_turns)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        last_active=excluded.last_active,
                        total_turns=excluded.total_turns
                """, (session_id, record.created_at, now, record.total_turns))

                # Insert turn
                conn.execute("""
                    INSERT INTO turns(session_id, turn_index, user, assistant, timestamp, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, turn_index, user, assistant, now, json.dumps(metadata)))

                # Log safety event if not safe
                if safety_level not in ("safe",):
                    conn.execute("""
                        INSERT INTO safety_events(session_id, level, triggered_by, matched_json, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (session_id, safety_level,
                          metadata.get("triggered_by", "unknown"),
                          json.dumps(metadata.get("matched", [])),
                          now))

        with self._lock:
            self._hot_cache[session_id] = record

        return updated_keys

    def log_phq(self, session_id: str, score: int, severity: str):
        """Record a PHQ-8 score."""
        now  = time.time()
        record = self.get_session(session_id)
        if record:
            record.phq_history.append(PHQRecord(score, severity, now))
        if self.backend == "sqlite":
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO phq_scores(session_id, score, severity, timestamp) VALUES (?,?,?,?)",
                    (session_id, score, severity, now)
                )

    def log_mood(self, session_id: str, mood: int, energy: int, sleep: int):
        """Record a mood check-in."""
        now = time.time()
        record = self.get_session(session_id)
        if record:
            record.mood_history.append(MoodRecord(mood, energy, sleep, now))
        if self.backend == "sqlite":
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO mood_logs(session_id, mood, energy, sleep, timestamp) VALUES (?,?,?,?,?)",
                    (session_id, mood, energy, sleep, now)
                )

    def set_goal(self, session_id: str, goal_text: str) -> str:
        """Create a therapeutic goal. Returns goal_id."""
        import uuid
        goal_id = str(uuid.uuid4())[:8]
        now     = time.time()
        goal    = GoalRecord(goal_id, goal_text, "active", now, now)
        record  = self.get_session(session_id)
        if record:
            record.goals.append(goal)
        if self.backend == "sqlite":
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO goals(goal_id, session_id, goal_text, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                    (goal_id, session_id, goal_text, "active", now, now)
                )
        return goal_id

    def update_goal_status(self, session_id: str, goal_id: str, status: str):
        now = time.time()
        record = self.get_session(session_id)
        if record:
            for g in record.goals:
                if g.goal_id == goal_id:
                    g.status     = status
                    g.updated_at = now
        if self.backend == "sqlite":
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "UPDATE goals SET status=?, updated_at=? WHERE goal_id=? AND session_id=?",
                    (status, now, goal_id, session_id)
                )

    def update_summary(self, session_id: str, summary: str):
        record = self.get_session(session_id)
        if record:
            record.summary = summary
        if self.backend == "sqlite":
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "UPDATE sessions SET summary=? WHERE session_id=?",
                    (summary, session_id)
                )

    def set_active_assessment(self, session_id: str, active: bool):
        record = self.get_session(session_id)
        if record:
            record.active_assessment = active
        if self.backend == "sqlite":
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "UPDATE sessions SET active_assessment=? WHERE session_id=?",
                    (int(active), session_id)
                )

    def get_progress_report(self, session_id: str) -> Dict:
        """Return structured progress data for a session."""
        record = self.get_session(session_id)
        if not record:
            return {"error": "Session not found"}

        phq_trend = [{"score": p.score, "severity": p.severity,
                      "date": datetime.fromtimestamp(p.timestamp).strftime("%Y-%m-%d")}
                     for p in record.phq_history]
        mood_trend = [{"mood": m.mood, "energy": m.energy, "sleep": m.sleep,
                       "date": datetime.fromtimestamp(m.timestamp).strftime("%Y-%m-%d")}
                      for m in record.mood_history]
        active_goals = [{"id": g.goal_id, "text": g.goal_text, "status": g.status}
                        for g in record.goals if g.status == "active"]

        return {
            "session_id":   session_id,
            "total_turns":  record.total_turns,
            "phq_history":  phq_trend,
            "mood_history": mood_trend,
            "active_goals": active_goals,
            "summary":      record.summary,
        }

    def session_count(self) -> int:
        if self.backend == "sqlite" and os.path.exists(self._db_path):
            with sqlite3.connect(self._db_path) as conn:
                return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return len(self._hot_cache)
