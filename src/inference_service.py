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

from ct_io import Skip, assemble_volume
from db import IMAGES_DIR, VOLUMES_DIR, connect, get_state, set_state
from gradcam import MODEL_VERSION, THRESHOLD, load_model, predict_with_heatmap
from preprocess import load_dicom

URL = os.getenv("ORTHANC_URL", "http://localhost:8042")
AUTH = (os.getenv("ORTHANC_USER", "demo"), os.getenv("ORTHANC_PASSWORD", "demo"))
VIEW_SIZE = 512
XRAY_MODALITIES = {"CR", "DX"}

_ct_model = None   # loaded on first use, so X-ray analysis works even without MONAI


def heat_rgba(heat, size=VIEW_SIZE):
    """Grad-CAM map -> transparent RGBA image (cold areas invisible), for the viewer overlay."""
    h = Image.fromarray((heat * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    h = np.asarray(h, dtype=np.float32) / 255.0
    rgb = matplotlib.colormaps["jet"](h)[..., :3]
    alpha = np.clip(h * 1.4, 0, 1) * 0.65
    return Image.fromarray((np.dstack([rgb, alpha]) * 255).astype(np.uint8), "RGBA")


def analyse_xray(session, iid, study_id, model, row):
    raw = session.get(f"{URL}/instances/{iid}/file", timeout=60).content
    score, _, heat = predict_with_heatmap(io.BytesIO(raw), model)
    flagged = score >= THRESHOLD
    scan = Image.fromarray(load_dicom(io.BytesIO(raw))).resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR)
    scan.save(IMAGES_DIR / f"{study_id}_scan.png")
    row["scan_path"] = f"images/{study_id}_scan.png"
    if flagged:   # a heatmap on a "normal" scan would mislead, so only draw it when flagged
        heat_rgba(heat).save(IMAGES_DIR / f"{study_id}_heatmap.png")
        row["heatmap_path"] = f"images/{study_id}_heatmap.png"
    row.update(score=float(score), flagged=int(flagged), threshold=THRESHOLD, model_version=MODEL_VERSION)


def read_ct_series(session, instances):
    """Download all slices of the study, keep the largest series, return slice dicts."""
    by_series = {}
    for inst in instances:
        raw = session.get(f"{URL}/instances/{inst['ID']}/file", timeout=60).content
        ds = pydicom.dcmread(io.BytesIO(raw))
        if "ImagePositionPatient" not in ds or "PixelData" not in ds:
            continue
        by_series.setdefault(str(ds.SeriesInstanceUID), []).append(dict(
            pix=ds.pixel_array,
            ipp=[float(v) for v in ds.ImagePositionPatient],
            iop=[float(v) for v in ds.ImageOrientationPatient],
            spacing=[float(v) for v in ds.PixelSpacing],
            slope=float(ds.get("RescaleSlope", 1)), intercept=float(ds.get("RescaleIntercept", 0))))
    if not by_series:
        raise Skip("no image slices found in this CT study")
    return max(by_series.values(), key=len)


def analyse_ct(session, instances, study_id, row):
    global _ct_model
    slices = read_ct_series(session, instances)
    volume, affine = assemble_volume(slices)          # raises Skip for unsuitable series
    from ct_model import CT_MODEL_VERSION, load_ct_model, segment   # heavy import, only when needed
    if _ct_model is None:
        print("  loading CT model ...")
        _ct_model = load_ct_model()
    res = segment(volume, affine, _ct_model)
    np.save(VOLUMES_DIR / f"{study_id}_ct.npy", res["ct"])
    np.save(VOLUMES_DIR / f"{study_id}_mask.npy", res["mask"])
    row.update(volume_path=study_id, n_slices=res["n_slices"], best_slice=res["best_slice"],
               spleen_volume_ml=res["volume_ml"], score=res["confidence"], model_version=CT_MODEL_VERSION)
    if res["confidence"] is None:
        row["error"] = "The model found no spleen in this volume."


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
               scan_path=None, heatmap_path=None, error=None, volume_path=None, n_slices=None,
               spleen_volume_ml=None, best_slice=None,
               processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    try:
        if modality in XRAY_MODALITIES:
            analyse_xray(session, iid, study_id, model, row)
        elif modality == "CT":
            analyse_ct(session, instances, study_id, row)
        else:
            raise Skip(f"No model for modality '{modality}' yet")
    except Skip as why:
        row["status"], row["error"] = "skipped", str(why)
    except Exception:
        row["status"], row["error"] = "error", traceback.format_exc()[-800:]

    conn.execute("""INSERT INTO studies (study_id, instance_id, patient_id, modality, patient_sex,
        patient_age, study_date, status, score, flagged, threshold, model_version, scan_path,
        heatmap_path, error, processed_at, volume_path, n_slices, spleen_volume_ml, best_slice)
        VALUES (:study_id, :instance_id, :patient_id, :modality, :patient_sex, :patient_age,
        :study_date, :status, :score, :flagged, :threshold, :model_version, :scan_path,
        :heatmap_path, :error, :processed_at, :volume_path, :n_slices, :spleen_volume_ml, :best_slice)""", row)
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
    # Safety net: also pick up studies we have not processed yet, but ONLY once Orthanc says they
    # are stable. A CT series arrives as hundreds of files, so a half-uploaded study must wait.
    for sid in session.get(f"{URL}/studies", timeout=30).json():
        if sid in ids or conn.execute("SELECT 1 FROM studies WHERE study_id=?", (sid,)).fetchone():
            continue
        if session.get(f"{URL}/studies/{sid}", timeout=30).json().get("IsStable"):
            ids.append(sid)
    results = []
    for sid in dict.fromkeys(ids):   # de-duplicate, keep order
        r = process_study(conn, model, session, sid)
        if r:
            results.append(r)
            shown = f"{r['score']:.2f}" if r["score"] is not None else "-"
            extra = f"  spleen={r['spleen_volume_ml']:.0f} mL" if r["spleen_volume_ml"] is not None else ""
            note = f"  ({r['error'][:70]})" if r["status"] != "done" and r["error"] else ""
            print(f"  {r['patient_id'][:12]:12s} {r['modality']:3s}  {r['status']:7s}  score={shown}{extra}{note}")
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
        try:
            n = len(run_cycle(conn, model, session))
        except requests.exceptions.RequestException as e:
            if args.once:
                raise
            print(f"Orthanc not reachable ({type(e).__name__}), retrying in 10 s ...")
            time.sleep(10)
            continue
        if args.once:
            print("Processed", n, "new studies")
            break
        time.sleep(args.interval)