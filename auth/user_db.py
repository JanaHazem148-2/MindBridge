"""
auth/user_db.py
────────────────
SQLite-backed user authentication for MindBridge.

Schema:
  users(user_id, phone_hash, password_hash, display_name, role,
        created_at, last_login_at, is_active)

Phone numbers are stored as SHA-256 hashes for privacy.
Passwords are hashed with bcrypt (or fallback to pbkdf2 if bcrypt unavailable).
"""

import os
import uuid
import sqlite3
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ── Try bcrypt, fall back to hashlib.pbkdf2 ──────────────────────────────────
try:
    import bcrypt
    _HAS_BCRYPT = True
    logger.info("UserDB: using bcrypt for password hashing")
except ImportError:
    _HAS_BCRYPT = False
    logger.warning("UserDB: bcrypt not installed — using pbkdf2_hmac fallback")


def _hash_password(password: str) -> str:
    if _HAS_BCRYPT:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    salt = os.urandom(32)
    dk   = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return salt.hex() + ":" + dk.hex()


def _check_password(password: str, stored: str) -> bool:
    if _HAS_BCRYPT and stored.startswith("$2"):
        return bcrypt.checkpw(password.encode(), stored.encode())
    # pbkdf2 fallback
    try:
        salt_hex, dk_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        dk   = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
        return dk.hex() == dk_hex
    except Exception:
        return False


def _hash_phone(phone: str) -> str:
    """Normalise and SHA-256 the phone for privacy-preserving lookups."""
    normalised = "".join(c for c in phone if c.isdigit())
    return hashlib.sha256(normalised.encode()).hexdigest()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id         TEXT PRIMARY KEY,
    phone_hash      TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    display_name    TEXT DEFAULT 'User',
    role            TEXT DEFAULT 'user',
    created_at      REAL NOT NULL,
    last_login_at   REAL,
    is_active       INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone_hash);

CREATE TABLE IF NOT EXISTS clinician_patients (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    clinician_id    TEXT NOT NULL,
    patient_id      TEXT NOT NULL,
    invite_code     TEXT UNIQUE NOT NULL,
    status          TEXT DEFAULT 'pending',   -- 'pending' | 'active' | 'revoked'
    linked_at       REAL,
    created_at      REAL NOT NULL,
    UNIQUE(clinician_id, patient_id)
);
CREATE INDEX IF NOT EXISTS idx_cp_clinician ON clinician_patients(clinician_id);
CREATE INDEX IF NOT EXISTS idx_cp_patient   ON clinician_patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_cp_code      ON clinician_patients(invite_code);
"""


class UserDB:
    """
    Thin SQLite wrapper for user accounts.

    Usage:
        db = UserDB(db_path="data/users.db")
        result = db.register(phone="+201001234567", password="secret", display_name="Ahmed")
        result = db.login(phone="+201001234567", password="secret")
    """

    def __init__(self, db_path: str = "./data/users.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        logger.info(f"UserDB initialised → {self.db_path}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
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

    def _ensure_schema(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ── Public API ─────────────────────────────────────────────────────────────

    def register(
        self,
        phone: str,
        password: str,
        display_name: str = "User",
        role: str = "user",
    ) -> Dict[str, Any]:
        """
        Register a new user.
        Returns: {"ok": True, "user_id": ..., "role": ...}
                 {"ok": False, "error": "..."}
        """
        if not phone or not password:
            return {"ok": False, "error": "Phone and password are required."}
        if len(password) < 6:
            return {"ok": False, "error": "Password must be at least 6 characters."}

        phone_hash  = _hash_phone(phone)
        pwd_hash    = _hash_password(password)
        user_id     = "u-" + uuid.uuid4().hex[:20]

        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO users
                       (user_id, phone_hash, password_hash, display_name, role, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (user_id, phone_hash, pwd_hash, display_name, role, time.time()),
                )
            logger.info(f"UserDB: registered user_id={user_id} role={role}")
            return {"ok": True, "user_id": user_id, "role": role}
        except sqlite3.IntegrityError:
            return {"ok": False, "error": "رقم الهاتف مسجّل بالفعل — Phone already registered."}
        except Exception as e:
            logger.exception("UserDB register error")
            return {"ok": False, "error": str(e)}

    def login(self, phone: str, password: str) -> Dict[str, Any]:
        """
        Verify credentials.
        Returns: {"ok": True, "user": {...}} or {"ok": False, "error": "..."}
        """
        if not phone or not password:
            return {"ok": False, "error": "Phone and password are required."}

        phone_hash = _hash_phone(phone)
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM users WHERE phone_hash=? AND is_active=1",
                    (phone_hash,)
                ).fetchone()

            if not row:
                return {"ok": False, "error": "رقم الهاتف غير مسجّل — Phone not found."}

            if not _check_password(password, row["password_hash"]):
                return {"ok": False, "error": "كلمة المرور غير صحيحة — Wrong password."}

            # Update last login
            with self._conn() as conn:
                conn.execute(
                    "UPDATE users SET last_login_at=? WHERE user_id=?",
                    (time.time(), row["user_id"])
                )

            user = {
                "user_id":      row["user_id"],
                "display_name": row["display_name"],
                "role":         row["role"],
            }
            logger.info(f"UserDB: login ok user_id={user['user_id']}")
            return {"ok": True, "user": user}
        except Exception as e:
            logger.exception("UserDB login error")
            return {"ok": False, "error": str(e)}

    def get_user(self, user_id: str) -> Optional[Dict]:
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT user_id, display_name, role, created_at FROM users WHERE user_id=?",
                    (user_id,)
                ).fetchone()
            if not row:
                return None
            return dict(row)
        except Exception:
            return None

    def update_password(self, user_id: str, new_password: str) -> bool:
        if len(new_password) < 6:
            return False
        try:
            pwd_hash = _hash_password(new_password)
            with self._conn() as conn:
                conn.execute(
                    "UPDATE users SET password_hash=? WHERE user_id=?",
                    (pwd_hash, user_id)
                )
            return True
        except Exception:
            return False

    def deactivate(self, user_id: str) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("UPDATE users SET is_active=0 WHERE user_id=?", (user_id,))
            return True
        except Exception:
            return False

    # ── Clinician-Patient relationship ─────────────────────────────────────────

    def create_invite(self, clinician_id: str) -> Dict[str, Any]:
        """Generate a one-time invite code for a clinician to share with a patient."""
        import secrets
        code = "MB-" + secrets.token_urlsafe(8).upper()
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO clinician_patients
                       (clinician_id, patient_id, invite_code, status, created_at)
                       VALUES (?, '', ?, 'pending', ?)""",
                    (clinician_id, code, time.time()),
                )
            logger.info(f"UserDB: invite created code={code} clinician={clinician_id}")
            return {"ok": True, "invite_code": code}
        except Exception as e:
            logger.exception("create_invite error")
            return {"ok": False, "error": str(e)}

    def accept_invite(self, patient_id: str, invite_code: str) -> Dict[str, Any]:
        """Patient redeems an invite code to link themselves to a clinician."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM clinician_patients WHERE invite_code=? AND status='pending'",
                    (invite_code,)
                ).fetchone()
                if not row:
                    return {"ok": False, "error": "Invalid or already used invite code."}
                if row["patient_id"] and row["patient_id"] != patient_id:
                    return {"ok": False, "error": "This code was issued for a different patient."}
                # Check not already linked
                existing = conn.execute(
                    "SELECT id FROM clinician_patients WHERE clinician_id=? AND patient_id=? AND status='active'",
                    (row["clinician_id"], patient_id)
                ).fetchone()
                if existing:
                    return {"ok": False, "error": "You are already linked to this clinician."}
                conn.execute(
                    """UPDATE clinician_patients
                       SET patient_id=?, status='active', linked_at=?
                       WHERE invite_code=?""",
                    (patient_id, time.time(), invite_code),
                )
                clinician_id = row["clinician_id"]
            logger.info(f"UserDB: patient {patient_id} linked to clinician {clinician_id}")
            return {"ok": True, "clinician_id": clinician_id}
        except Exception as e:
            logger.exception("accept_invite error")
            return {"ok": False, "error": str(e)}

    def get_my_patients(self, clinician_id: str) -> List[Dict]:
        """Return all active patients linked to a clinician."""
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT cp.patient_id, cp.linked_at, u.display_name, u.created_at as member_since
                       FROM clinician_patients cp
                       JOIN users u ON u.user_id = cp.patient_id
                       WHERE cp.clinician_id=? AND cp.status='active'
                       ORDER BY cp.linked_at DESC""",
                    (clinician_id,)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.exception("get_my_patients error")
            return []

    def get_my_clinician(self, patient_id: str) -> Optional[Dict]:
        """Return the clinician linked to a patient (if any)."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    """SELECT cp.clinician_id, cp.linked_at, u.display_name
                       FROM clinician_patients cp
                       JOIN users u ON u.user_id = cp.clinician_id
                       WHERE cp.patient_id=? AND cp.status='active'
                       LIMIT 1""",
                    (patient_id,)
                ).fetchone()
            return dict(row) if row else None
        except Exception:
            return None

    def is_my_patient(self, clinician_id: str, patient_id: str) -> bool:
        """Check if a patient is actively linked to this clinician."""
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT 1 FROM clinician_patients WHERE clinician_id=? AND patient_id=? AND status='active'",
                    (clinician_id, patient_id)
                ).fetchone()
            return row is not None
        except Exception:
            return False

    def revoke_link(self, clinician_id: str, patient_id: str) -> bool:
        """Clinician removes a patient from their list."""
        try:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE clinician_patients SET status='revoked' WHERE clinician_id=? AND patient_id=?",
                    (clinician_id, patient_id)
                )
            return True
        except Exception:
            return False
