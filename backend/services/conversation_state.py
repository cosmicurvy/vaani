from __future__ import annotations

import json
import sqlite3
from pathlib import Path

class ConversationState:
    def __init__(self, db_path=None):
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = (Path(__file__).resolve().parents[2]
                            / "data"
                            / "vaani_support.db")
        self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.initialize()

    def connect(self):
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self):
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_states (

                    session_id TEXT PRIMARY KEY,

                    pending_action TEXT,

                    context TEXT NOT NULL DEFAULT '{}',

                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get_state(self, session_id):
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM conversation_states
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if not row:
            return {
                "pending_action": None,
                "context": {}
            }

        return {
            "pending_action": row["pending_action"],
            "context": json.loads(row["context"])
        }

    def set_state(
        self,
        session_id,
        pending_action,
        context=None):
        context_json = json.dumps(context or {})
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_states
                    (session_id, pending_action, context, updated_at)

                VALUES (?, ?, ?, CURRENT_TIMESTAMP)

                ON CONFLICT(session_id)
                DO UPDATE SET

                    pending_action = excluded.pending_action,

                    context = excluded.context,

                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    session_id,
                    pending_action,
                    context_json
                )
            )

    def clear_state(self, session_id):
        with self.connect() as conn:
            conn.execute(
                """
                DELETE FROM conversation_states
                WHERE session_id = ?
                """,
                (session_id,),)