import os
import sqlite3
from pathlib import Path

RUNTIME = Path(os.getenv("RUNTIME_DIR", "runtime"))
DB_PATH = RUNTIME / "app.db"
IMAGES_DIR = RUNTIME / "images"
VOLUMES_DIR = RUNTIME / "volumes"

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    study_id          TEXT PRIMARY KEY,   -- Orthanc study ID
    instance_id       TEXT,
    patient_id        TEXT,
    modality          TEXT,
    patient_sex       TEXT,
    patient_age       TEXT,
    study_date        TEXT,
    status            TEXT,               -- done | skipped | error
    score             REAL,               -- X-ray: AI score. CT: mean model probability inside the mask
    flagged           INTEGER,            -- X-ray only
    threshold         REAL,               -- X-ray only
    model_version     TEXT,
    scan_path         TEXT,
    heatmap_path      TEXT,
    error             TEXT,               -- reason when skipped / failed
    review_status     TEXT DEFAULT 'pending',  -- pending | approved | rejected
    processed_at      TEXT,
    volume_path       TEXT,               -- CT only: name of the stored 3D arrays
    n_slices          INTEGER,            -- CT only
    spleen_volume_ml  REAL,               -- CT only
    best_slice        INTEGER             -- CT only: slice with the largest predicted area
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

# Columns added after the first version. Older databases get them automatically.
MIGRATIONS = {
    "volume_path": "TEXT",
    "n_slices": "INTEGER",
    "spleen_volume_ml": "REAL",
    "best_slice": "INTEGER",
}

_ready = set()   # database files whose schema has already been created in this process


def _migrate(conn):
    have = {r["name"] for r in conn.execute("PRAGMA table_info(studies)")}
    for name, col_type in MIGRATIONS.items():
        if name not in have:
            try:
                conn.execute(f"ALTER TABLE studies ADD COLUMN {name} {col_type}")
            except sqlite3.OperationalError:   # another process added it first
                pass
    conn.commit()


def connect():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VOLUMES_DIR.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI may use the connection from another worker thread.
    # Each request gets its own connection, so this is safe.
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    key = str(DB_PATH.resolve())
    if key not in _ready:        # create the schema once, not on every request
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        _migrate(conn)
        _ready.add(key)
    return conn


def get_state(conn, key, default=None):
    row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_state(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO state(key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()