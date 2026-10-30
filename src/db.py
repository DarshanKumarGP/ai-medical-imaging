import os
import sqlite3
from pathlib import Path

RUNTIME = Path(os.getenv("RUNTIME_DIR", "runtime"))
DB_PATH = RUNTIME / "app.db"
IMAGES_DIR = RUNTIME / "images"

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    study_id      TEXT PRIMARY KEY,   -- Orthanc study ID
    instance_id   TEXT,
    patient_id    TEXT,
    modality      TEXT,
    patient_sex   TEXT,
    patient_age   TEXT,
    study_date    TEXT,
    status        TEXT,               -- done | skipped | error
    score         REAL,               -- AI confidence score (not a calibrated probability)
    flagged       INTEGER,
    threshold     REAL,
    model_version TEXT,
    scan_path     TEXT,
    heatmap_path  TEXT,
    error         TEXT,
    review_status TEXT DEFAULT 'pending',  -- pending | approved | rejected
    processed_at  TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id      TEXT NOT NULL,
    reviewer      TEXT NOT NULL,
    decision      TEXT NOT NULL,
    comment       TEXT,
    ai_score      REAL,
    flagged       INTEGER,
    model_version TEXT,
    created_at    TEXT NOT NULL
);

-- The audit log is append-only: the database itself refuses edits and deletes.
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;

CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


def connect():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def get_state(conn, key, default=None):
    row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_state(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO state(key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()