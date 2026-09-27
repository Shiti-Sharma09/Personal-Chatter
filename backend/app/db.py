"""Session/message persistence via the stdlib sqlite3 module.

Chosen deliberately over a server-based DB or an ORM: this is a
single-user local app, so a file-based store with zero extra processes
and zero migration tooling to maintain is the right amount of
infrastructure. SQLite handles that with no extra dependency at all.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    conn = sqlite3.connect(config.SESSIONS_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.executescript(SCHEMA)


def create_session(title="New chat"):
    session_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
    return {"id": session_id, "title": title, "created_at": now, "updated_at": now}


def list_sessions():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_session(session_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def rename_session(session_id, title):
    with _connect() as conn:
        conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, _now(), session_id))


def touch_session(session_id):
    with _connect() as conn:
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))


def delete_session(session_id):
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def add_message(session_id, role, content, sources=None):
    now = _now()
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(sources or []), now),
        )
        message_id = cursor.lastrowid
    touch_session(session_id)
    return {
        "id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "created_at": now,
    }


def list_messages(session_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
        ).fetchall()
    messages = []
    for row in rows:
        message = dict(row)
        message["sources"] = json.loads(message["sources"])
        messages.append(message)
    return messages
