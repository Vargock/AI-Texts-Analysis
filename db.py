# db.py
import sqlite3
import logging
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)
DB_PATH = "data.db"

# Module-level cached connection
_conn: Optional[sqlite3.Connection] = None


def get_db_connection() -> Optional[sqlite3.Connection]:
    global _conn
    if _conn:
        return _conn

    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=20)
        conn.row_factory = sqlite3.Row
        _conn = conn
        return conn
    except sqlite3.Error as e:
        logger.error("Database connection error: %s", e)
        return None


def init_db() -> None:
    conn = get_db_connection()

    if not conn:
        raise RuntimeError("Could not obtain database connection")

    try:
        cur = conn.cursor()
        # Enable WAL for better concurrency
        cur.execute("PRAGMA journal_mode=WAL")
        # Create table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Helpful indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON history(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON history(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_timestamp ON history(user_id, timestamp)")
        conn.commit()
        logger.info("Database initialized")

    except sqlite3.Error as e:
        logger.exception("Database initialization error: %s", e)
        raise


def save_message(user_id: int, text: str, sentiment: str, confidence: float) -> bool:
    conn = get_db_connection()
    if not conn:
        return False

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO history (user_id, text, sentiment, confidence) VALUES (?, ?, ?, ?)",
            (user_id, text, sentiment, confidence),
        )
        conn.commit()
        logger.debug("Saved message for user %s", user_id)
        return True
    except sqlite3.Error as e:
        logger.exception("Error saving message: %s", e)
        # rollback not strictly necessary after exception, but safe
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def get_recent(user_id: int, limit: int = 10) -> List[Tuple]:
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT text, sentiment, confidence, timestamp
            FROM history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
        return [tuple(r) for r in rows]
    except sqlite3.Error as e:
        logger.exception("Error fetching recent messages: %s", e)
        return []
