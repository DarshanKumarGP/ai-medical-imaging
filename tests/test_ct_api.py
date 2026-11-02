import io
import os
import sys
import tempfile
from pathlib import Path

# Use a throw-away database. This must happen before importing the app.
os.environ["RUNTIME_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import db
from api import app


@pytest.fixture(scope="module")
def client():
    conn = db.connect()
    ct = np.zeros((30, 24, 10), dtype=np.uint8)
    ct[10:20, 8:16, 4] = 180
    mask = np.zeros_like(ct)
    mask[10:20, 8:16, 4] = 1
    np.save(db.VOLUMES_DIR / "ct1_ct.npy", ct)
    np.save(db.VOLUMES_DIR / "ct1_mask.npy", mask)
    cols = ("study_id, patient_id, modality, status, score, model_version, volume_path, n_slices, "
            "spleen_volume_ml, best_slice, error, processed_at")
    rows = [
        ("ct1", "spleen_1", "CT", "done", 0.97, "monai-unet3d-spleen-v1", "ct1", 10, 12.3, 4, None, "2026-01-01"),
        ("ct2", "tiny", "CT", "skipped", None, None, None, None, None, None,
         "CT series has only 1 slice(s); 3D segmentation needs at least 30", "2026-01-01"),
    ]
    conn.executemany(f"INSERT INTO studies ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return TestClient(app)


def test_ct_detail_fields(client):
    d = client.get("/studies/ct1").json()
    assert d["kind"] == "ct" and d["n_slices"] == 10 and d["best_slice"] == 4
    assert d["spleen_volume_ml"] == 12.3 and d["status"] == "done"
    assert "volume_path" not in d            # internal file names are not exposed


def test_ct_slice_images(client):
    r = client.get("/studies/ct1/slice/4/scan")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert "max-age" in r.headers["cache-control"]
    scan = Image.open(io.BytesIO(r.content))
    assert scan.size == (512, 512) and scan.mode == "L"
    mask = Image.open(io.BytesIO(client.get("/studies/ct1/slice/4/mask").content))
    assert mask.mode == "RGBA" and np.asarray(mask)[..., 3].max() > 0
    empty = Image.open(io.BytesIO(client.get("/studies/ct1/slice/0/mask").content))
    assert np.asarray(empty)[..., 3].max() == 0          # no spleen on this slice


def test_slice_out_of_range_and_missing_volume(client):
    assert client.get("/studies/ct1/slice/10/scan").status_code == 404
    assert client.get("/studies/ct1/slice/-1/scan").status_code == 404
    assert client.get("/studies/ct2/slice/0/scan").status_code == 404     # skipped study has no volume
    assert client.get("/studies/nope/slice/0/scan").status_code == 404


def test_ct_review_goes_to_audit_log(client):
    r = client.post("/studies/ct1/decision", json={"reviewer": "Dr. CT", "decision": "approved"})
    assert r.status_code == 201
    entry = client.get("/studies/ct1").json()["audit"][0]
    assert entry["ai_score"] == 0.97 and entry["flagged"] is None
    assert entry["model_version"] == "monai-unet3d-spleen-v1"


def test_skipped_ct_explains_why_and_cannot_be_reviewed(client):
    d = client.get("/studies/ct2").json()
    assert d["status"] == "skipped" and "at least 30" in d["error"]
    r = client.post("/studies/ct2/decision", json={"reviewer": "X", "decision": "approved"})
    assert r.status_code == 409