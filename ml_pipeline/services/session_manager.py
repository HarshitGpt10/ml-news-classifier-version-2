"""
session_manager.py
───────────────────
SQLite-backed session store.  Persists:
  • Every uploaded article / text (with translation + classification result)
  • Every chatbot Q&A exchange
  • Session metadata (timestamps, language stats)

No external database needed — uses Python's built-in sqlite3.

Usage:
    from ml_pipeline.services.session_manager import SessionManager
    sm = SessionManager()
    session_id = sm.create_session()
    sm.add_article(session_id, article_record)
    sm.add_chat_message(session_id, role="user", content="What was the main topic?")
    history = sm.get_session(session_id)
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("ml_pipeline/data/sessions.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class SessionManager:

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_db()

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          TEXT PRIMARY KEY,
                    name        TEXT,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS articles (
                    id              TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL REFERENCES sessions(id),
                    original_text   TEXT NOT NULL,
                    translated_text TEXT,
                    source_lang     TEXT,
                    was_translated  INTEGER DEFAULT 0,
                    category        TEXT,
                    label           INTEGER,
                    confidence      REAL,
                    probabilities   TEXT,    -- JSON
                    word_count      INTEGER,
                    source_type     TEXT,    -- "text" | "image"
                    image_filename  TEXT,
                    ocr_confidence  REAL,
                    created_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id          TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL REFERENCES sessions(id),
                    article_id  TEXT REFERENCES articles(id),
                    role        TEXT NOT NULL,   -- "user" | "assistant"
                    content     TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_art_session  ON articles(session_id);
                CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_chat_article ON chat_messages(article_id);
            """)

    # ── Session CRUD ──────────────────────────────────────────────────────────

    def create_session(self, name: str = "") -> str:
        sid = str(uuid.uuid4())
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?)",
                (sid, name or f"Session {now[:10]}", now, now),
            )
        return sid

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not row:
                return None
            articles = conn.execute(
                "SELECT * FROM articles WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()
            messages = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()

        return {
            "id":         row["id"],
            "name":       row["name"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "articles":   [_row_to_article(a) for a in articles],
            "messages":   [dict(m) for m in messages],
            "stats":      _compute_stats([_row_to_article(a) for a in articles]),
        }

    def list_sessions(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT s.id, s.name, s.created_at, s.updated_at,
                          COUNT(DISTINCT a.id) AS article_count,
                          COUNT(DISTINCT c.id) AS message_count
                   FROM sessions s
                   LEFT JOIN articles     a ON a.session_id = s.id
                   LEFT JOIN chat_messages c ON c.session_id = s.id
                   GROUP BY s.id
                   ORDER BY s.updated_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM articles WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # ── Article CRUD ──────────────────────────────────────────────────────────

    def add_article(self, session_id: str, data: dict) -> str:
        """data keys: original_text, translated_text, source_lang, was_translated,
                      category, label, confidence, probabilities, word_count,
                      source_type, image_filename, ocr_confidence"""
        # Auto-create session if it doesn't exist
        with self._conn() as conn:
            if not conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone():
                now = _now()
                conn.execute("INSERT INTO sessions VALUES (?,?,?,?)",
                             (session_id, f"Session {now[:10]}", now, now))

        aid = str(uuid.uuid4())
        now = _now()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO articles VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                aid,
                session_id,
                data.get("original_text", ""),
                data.get("translated_text", ""),
                data.get("source_lang", "en"),
                int(data.get("was_translated", False)),
                data.get("category", ""),
                data.get("label", -1),
                data.get("confidence", 0.0),
                json.dumps(data.get("probabilities", {})),
                data.get("word_count", 0),
                data.get("source_type", "text"),
                data.get("image_filename", ""),
                data.get("ocr_confidence", None),
                now,
            ))
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id)
            )
        return aid

    def get_article(self, article_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM articles WHERE id=?", (article_id,)
            ).fetchone()
        return _row_to_article(row) if row else None

    def get_session_articles(self, session_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM articles WHERE session_id=? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [_row_to_article(r) for r in rows]

    # ── Chat CRUD ─────────────────────────────────────────────────────────────

    def add_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        article_id: str = "",
    ) -> str:
        mid = str(uuid.uuid4())
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_messages VALUES (?,?,?,?,?,?)",
                (mid, session_id, article_id or None, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id)
            )
        return mid

    def get_chat_history(
        self, session_id: str, article_id: str = "", limit: int = 50
    ) -> list[dict]:
        with self._conn() as conn:
            if article_id:
                rows = conn.execute(
                    """SELECT * FROM chat_messages
                       WHERE session_id=? AND article_id=?
                       ORDER BY created_at DESC LIMIT ?""",
                    (session_id, article_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM chat_messages
                       WHERE session_id=?
                       ORDER BY created_at DESC LIMIT ?""",
                    (session_id, limit),
                ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Context builder for chatbot ───────────────────────────────────────────

    def build_context(self, session_id: str, article_id: str = "") -> str:
        """Build a context string from session articles for the chatbot."""
        articles = (
            [self.get_article(article_id)]
            if article_id
            else self.get_session_articles(session_id)
        )
        articles = [a for a in articles if a]
        if not articles:
            return ""

        parts = []
        for i, art in enumerate(articles[-5:], 1):   # last 5 articles for context
            parts.append(
                f"[Article {i}] Category: {art['category']} | "
                f"Language: {art['source_lang']}\n"
                f"{art['translated_text'] or art['original_text']}"
            )
        return "\n\n".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


def _row_to_article(row) -> dict:
    d = dict(row)
    try:
        d["probabilities"] = json.loads(d.get("probabilities") or "{}")
    except Exception:
        d["probabilities"] = {}
    d["was_translated"] = bool(d.get("was_translated"))
    return d


def _compute_stats(articles: list[dict]) -> dict:
    from collections import Counter
    cats   = Counter(a["category"] for a in articles if a.get("category"))
    langs  = Counter(a["source_lang"] for a in articles if a.get("source_lang"))
    return {
        "total":          len(articles),
        "by_category":    dict(cats.most_common()),
        "by_language":    dict(langs.most_common()),
        "translated_pct": round(
            sum(1 for a in articles if a.get("was_translated")) / max(len(articles), 1) * 100, 1
        ),
    }
