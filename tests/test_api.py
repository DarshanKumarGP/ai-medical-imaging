import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# Use a throw-away database. This must happen before importing the app.
os.environ["RUNTIME_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import db
from api import app


@pytest.fixture(scope="module")
def client():
    conn = db.connect()
    Image.new("L", (8, 8)).save(db.IMAGES_DIR / "s1_scan.png")
    cols = ("study_id, patient_id, modality, status, score, flagged, threshold, "
            "model_version, scan_path, heatmap_path, processed_at")
    rows = [
        ("s1", "p1", "CR", "done", 0.90, 1, 0.48, "test-v1", "images/s1_scan.png", None, "2026-01-01"),
        ("s2", "p2", "CT", "skipped", None, None, None, None, None, None, "2026-01-01"),
        ("s3", "p3", "CR", "done", 0.70, 1, 0.48, "test-v1", None, None, "2026-01-01"),
    ]
    conn.executemany(f"INSERT INTO studies ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and "NOT FOR CLINICAL USE" in r.json()["disclaimer"]


def test_list_and_filter(client):
    body = client.get("/studies").json()
    scores = [s["score"] for s in body["studies"] if s["score"] is not None]
    assert scores == sorted(scores, reverse=True)           # highest AI score first
    assert body["studies"][-1]["study_id"] == "s2"          # unscored study last
    flagged = client.get("/studies", params={"flagged": True}).json()["studies"]
    assert {s["study_id"] for s in flagged} == {"s1", "s3"}


def test_detail_and_404(client):
    r = client.get("/studies/s1").json()
    assert r["study_id"] == "s1" and r["flagged"] is True and "disclaimer" in r
    assert client.get("/studies/nope").status_code == 404


def test_scan_image_served_heatmap_404(client):
    r = client.get("/studies/s1/scan")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get("/studies/s1/heatmap").status_code == 404   # none stored for s1


def test_approve_writes_audit_entry(client):
    r = client.post("/studies/s1/decision",
                    json={"reviewer": "Dr. Test", "decision": "approved", "comment": "ok"})
    assert r.status_code == 201
    d = client.get("/studies/s1").json()
    assert d["review_status"] == "approved" and len(d["audit"]) == 1
    entry = d["audit"][0]
    assert entry["reviewer"] == "Dr. Test" and entry["ai_score"] == 0.9
    assert entry["model_version"] == "test-v1" and entry["created_at"]


def test_re_review_keeps_history(client):
    client.post("/studies/s3/decision", json={"reviewer": "A", "decision": "approved"})
    client.post("/studies/s3/decision", json={"reviewer": "B", "decision": "rejected"})
    d = client.get("/studies/s3").json()
    assert d["review_status"] == "rejected"
    assert [a["decision"] for a in d["audit"]] == ["approved", "rejected"]   # nothing overwritten


def test_invalid_input_rejected(client):
    assert client.post("/studies/s1/decision", json={"reviewer": "X", "decision": "maybe"}).status_code == 422
    assert client.post("/studies/s1/decision", json={"reviewer": "   ", "decision": "approved"}).status_code == 422
    assert client.post("/studies/nope/decision", json={"reviewer": "X", "decision": "approved"}).status_code == 404


def test_skipped_study_cannot_be_reviewed(client):
    r = client.post("/studies/s2/decision", json={"reviewer": "X", "decision": "approved"})
    assert r.status_code == 409


def test_audit_log_is_append_only():
    conn = db.connect()
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE audit_log SET decision='rejected'")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM audit_log")
    finally:
        conn.close()


def test_no_delete_or_edit_endpoints(client):
    assert client.delete("/audit").status_code == 405
    assert client.put("/audit").status_code == 405
    assert client.delete("/studies/s1").status_code == 405