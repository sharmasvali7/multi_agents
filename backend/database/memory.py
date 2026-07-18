"""
Conversation memory.

Uses SQLite by default (zero setup, file-based) so the project runs
immediately without installing MongoDB or Postgres. Swap `DB_URL` in .env
to a Postgres/MongoDB connection string + adjust this module if you want
to match the original spec's suggested stack for your submission.
"""
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.environ.get("SQLITE_PATH", os.path.join(os.path.dirname(__file__), "conversations.db"))


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                intents TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def new_session_id() -> str:
    return str(uuid.uuid4())


def save_message(session_id: str, role: str, content: str, intents: str = ""):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, intents, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), session_id, role, content, intents, datetime.utcnow().isoformat()),
        )
        conn.commit()


def get_history(session_id: str, limit: int = 10) -> str:
    """Return the last `limit` messages for a session as a formatted string."""
    with _connect() as conn:
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit),
        )
        rows = list(reversed(cur.fetchall()))
    return "\n".join(f"{role}: {content}" for role, content in rows)


def get_all_sessions() -> list:
    with _connect() as conn:
        cur = conn.execute("SELECT DISTINCT session_id FROM messages")
        return [row[0] for row in cur.fetchall()]
