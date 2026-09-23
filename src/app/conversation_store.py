"""SQLite-backed conversation history.

One local file (data/processed/conversations.db) holds every conversation
and message. sqlite3 is part of the Python standard library, so this adds
zero new dependencies. A conversation is created lazily, on the first
message sent — starting a "New conversation" in the UI doesn't write
anything until the user actually asks something.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from src.config import CONFIG, project_path

DB_PATH = project_path(CONFIG["paths"].get("conversations_db", "data/processed/conversations.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    author_key TEXT NOT NULL,
    mode TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_conversation(author_key: str, mode: str, title: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, author_key, mode, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (title[:80], author_key, mode, _now(), _now()),
        )
        return cur.lastrowid


def add_message(conversation_id: int, role: str, content: str, sources: list[dict] | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, json.dumps(sources or []), _now()),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id))


def list_conversations() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, author_key, mode, created_at, updated_at "
            "FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [
        {"id": r[0], "title": r[1], "author_key": r[2], "mode": r[3], "created_at": r[4], "updated_at": r[5]}
        for r in rows
    ]


def get_messages(conversation_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, sources_json FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    return [{"role": r[0], "content": r[1], "sources": json.loads(r[2] or "[]")} for r in rows]


def delete_conversation(conversation_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))