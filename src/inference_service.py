import argparse
import io
import os
import time
import traceback
from datetime import datetime, timezone

import matplotlib
import numpy as np
import pydicom
import requests
from PIL import Image

from db import IMAGES_DIR, connect, get_state, set_state
from gradcam import MODEL_VERSION, THRESHOLD, load_model, predict_with_heatmap
from preprocess import load_dicom

URL = os.getenv("ORTHANC_URL", "http://localhost:8042")
AUTH = (os.getenv("ORTHANC_USER", "demo"), os.getenv("ORTHANC_PASSWORD", "demo"))
VIEW_SIZE = 512
XRAY_MODALITIES = {"CR", "DX"}


def heat_rgba(heat, size=VIEW_SIZE):
    """Grad-CAM map -> transparent RGBA image (cold areas invisible), for the viewer overlay."""
    h = Image.fromarray((heat * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    h = np.asarray(h, dtype=np.float32) / 255.0
    rgb = matplotlib.colormaps["jet"](h)[..., :3]
    alpha = np.clip(h * 1.4, 0, 1) * 0.65
    return Image.fromarray((np.dstack([rgb, alpha]) * 255).astype(np.uint8), "RGBA")


def process_study(conn, model, session, study_id):
    """Pull one study from Orthanc, run the AI, store the result. Skips known studies."""
    if conn.execute("SELECT 1 FROM studies WHERE study_id=?", (study_id,)).fetchone():
        return None
    instances = session.get(f"{URL}/studies/{study_id}/instances", timeout=30).json()
    iid = instances[0]["ID"]
    tags = session.get(f"{URL}/instances/{iid}/tags", params={"simplify": "true"}, timeout=30).json()
    modality = tags.get("Modality", "")

    row = dict(study_id=study_id, instance_id=iid, patient_id=tags.get("PatientID", ""),
               modality=modality, patient_sex=tags.get("PatientSex", ""),
               patient_age=tags.get("PatientAge", ""), study_date=tags.get("StudyDate", ""),
               status="done", score=None, flagged=None, threshold=None, model_version=None,
               scan_path=None, heatmap_path=None, error=None,
               processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    try:
        if modality in XRAY_MODALITIES:
            raw = session.get(f"{URL}/instances/{iid}/file", timeout=60).content
            score, _, heat = predict_with_heatmap(io.BytesIO(raw), model)
            flagged = score >= THRESHOLD
            scan = Image.fromarray(load_dicom(io.BytesIO(raw))).resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR)
            scan.save(IMAGES_DIR / f"{study_id}_scan.png")
            row["scan_path"] = f"images/{study_id}_scan.png"
            if flagged:   # a heatmap on a "normal" scan would mislead, so only draw it when flagged
                heat_rgba(heat).save(IMAGES_DIR / f"{study_id}_heatmap.png")
                row["heatmap_path"] = f"images/{study_id}_heatmap.png"
            row.update(score=float(score), flagged=int(flagged), threshold=THRESHOLD,
                       model_version=MODEL_VERSION)
        else:
            row["status"] = "skipped"
            row["error"] = f"No model for modality '{modality}' yet"
    except Exception:
        row["status"], row["error"] = "error", traceback.format_exc()[-800:]

    conn.execute("""INSERT INTO studies (study_id, instance_id, patient_id, modality, patient_sex,
        patient_age, study_date, status, score, flagged, threshold, model_version, scan_path,
        heatmap_path, error, processed_at) VALUES (:study_id, :instance_id, :patient_id, :modality,
        :patient_sex, :patient_age, :study_date, :status, :score, :flagged, :threshold,
        :model_version, :scan_path, :heatmap_path, :error, :processed_at)""", row)
    conn.commit()
    return row


def stable_study_ids(conn, session):
    """Ask Orthanc what changed since last time; return IDs of newly stable studies."""
    since = int(get_state(conn, "orthanc_seq", "0"))
    ids = []
    while True:
        data = session.get(f"{URL}/changes", params={"since": since, "limit": 100}, timeout=30).json()
        ids += [c["ID"] for c in data["Changes"] if c["ChangeType"] == "StableStudy"]
        since = data["Last"]
        if data["Done"]:
            break
    return ids, since


def run_cycle(conn, model, session):
    ids, since = stable_study_ids(conn, session)
    # Safety net: also pick up any study in Orthanc we have not processed yet
    ids += session.get(f"{URL}/studies", timeout=30).json()
    results = []
    for sid in dict.fromkeys(ids):   # de-duplicate, keep order
        r = process_study(conn, model, session, sid)
        if r:
            results.append(r)
            shown = f"{r['score']:.2f}" if r["score"] is not None else "-"
            print(f"  {r['patient_id'][:8]}  {r['modality']:3s}  {r['status']:7s}  score={shown}")
    set_state(conn, "orthanc_seq", since)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="process what is there, then exit")
    ap.add_argument("--interval", type=int, default=5)
    args = ap.parse_args()

    conn = connect()
    model = load_model()
    session = requests.Session()
    session.auth = AUTH
    print("Inference service started, watching", URL)
    while True:
        n = len(run_cycle(conn, model, session))
        if args.once:
            print("Processed", n, "new studies")
            break
        time.sleep(args.interval)