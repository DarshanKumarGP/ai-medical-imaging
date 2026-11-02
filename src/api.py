import os
from datetime import datetime, timezone
from typing import Literal, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

from ct_render import render_mask, render_scan
from db import RUNTIME, VOLUMES_DIR, connect

DISCLAIMER = "FOR EDUCATION ONLY. NOT FOR CLINICAL USE."
CACHE = {"Cache-Control": "private, max-age=3600"}   # study images never change once written

app = FastAPI(
    title="AI-Assisted Medical Imaging API",
    description=f"**{DISCLAIMER}** Lists AI-analysed studies (chest X-ray classification with Grad-CAM, "
                "CT spleen segmentation) and records radiologist decisions in an append-only audit log. "
                "The X-ray AI score is a risk score, not a calibrated probability.",
    version="1.1.0",
)
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def get_conn():
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def to_study(row):
    d = dict(row)
    sid = d["study_id"]
    d["flagged"] = None if d["flagged"] is None else bool(d["flagged"])
    d["scan_url"] = f"/studies/{sid}/scan" if d.pop("scan_path") else None
    d["heatmap_url"] = f"/studies/{sid}/heatmap" if d.pop("heatmap_path") else None
    d["kind"] = "ct" if d["modality"] == "CT" else "xray"
    d.pop("volume_path", None)
    d.pop("instance_id", None)
    return d


def get_study_or_404(conn, study_id):
    row = conn.execute("SELECT * FROM studies WHERE study_id=?", (study_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Study not found")
    return row


class Decision(BaseModel):
    reviewer: str = Field(min_length=1, max_length=80)
    decision: Literal["approved", "rejected"]
    comment: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("reviewer")
    @classmethod
    def reviewer_not_blank(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("reviewer must not be blank")
        return v


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.get("/studies")
def list_studies(
    review_status: Optional[Literal["pending", "approved", "rejected"]] = None,
    flagged: Optional[bool] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    conn=Depends(get_conn),
):
    where, args = [], []
    if review_status:
        where.append("review_status=?"); args.append(review_status)
    if flagged is not None:
        where.append("flagged=?"); args.append(int(flagged))
    sql = "SELECT * FROM studies"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY (score IS NULL), score DESC LIMIT ? OFFSET ?"
    rows = conn.execute(sql, args + [limit, offset]).fetchall()
    return {"disclaimer": DISCLAIMER, "count": len(rows), "studies": [to_study(r) for r in rows]}


@app.get("/studies/{study_id}")
def study_detail(study_id: str, conn=Depends(get_conn)):
    study = to_study(get_study_or_404(conn, study_id))
    audit = conn.execute("SELECT * FROM audit_log WHERE study_id=? ORDER BY id", (study_id,)).fetchall()
    return {"disclaimer": DISCLAIMER, **study, "audit": [dict(a) for a in audit]}


def _image(conn, study_id, column):
    path = get_study_or_404(conn, study_id)[column]
    if not path or not (RUNTIME / path).exists():
        raise HTTPException(404, "Image not available for this study")
    return FileResponse(RUNTIME / path, media_type="image/png", headers=CACHE)


@app.get("/studies/{study_id}/scan")
def study_scan(study_id: str, conn=Depends(get_conn)):
    return _image(conn, study_id, "scan_path")


@app.get("/studies/{study_id}/heatmap")
def study_heatmap(study_id: str, conn=Depends(get_conn)):
    return _image(conn, study_id, "heatmap_path")


def _open_volume(conn, study_id, z):
    """Memory-map the stored CT and mask arrays and check the slice number."""
    row = get_study_or_404(conn, study_id)
    if not row["volume_path"]:
        raise HTTPException(404, "No 3D volume for this study")
    ct_file = VOLUMES_DIR / f"{row['volume_path']}_ct.npy"
    mask_file = VOLUMES_DIR / f"{row['volume_path']}_mask.npy"
    if not ct_file.exists() or not mask_file.exists():
        raise HTTPException(404, "Volume files are missing")
    ct, mask = np.load(ct_file, mmap_mode="r"), np.load(mask_file, mmap_mode="r")
    if not 0 <= z < ct.shape[2]:
        raise HTTPException(404, f"Slice {z} out of range (0 to {ct.shape[2] - 1})")
    return ct, mask


@app.get("/studies/{study_id}/slice/{z}/scan")
def ct_slice_scan(study_id: str, z: int, conn=Depends(get_conn)):
    ct, _ = _open_volume(conn, study_id, z)
    return Response(render_scan(ct[:, :, z]), media_type="image/png", headers=CACHE)


@app.get("/studies/{study_id}/slice/{z}/mask")
def ct_slice_mask(study_id: str, z: int, conn=Depends(get_conn)):
    _, mask = _open_volume(conn, study_id, z)
    return Response(render_mask(mask[:, :, z]), media_type="image/png", headers=CACHE)


@app.post("/studies/{study_id}/decision", status_code=201)
def post_decision(study_id: str, body: Decision, conn=Depends(get_conn)):
    study = get_study_or_404(conn, study_id)
    if study["status"] != "done":
        raise HTTPException(409, f"Study has status '{study['status']}' and cannot be reviewed")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with conn:   # one transaction: the log entry and the status change succeed or fail together
        cur = conn.execute(
            """INSERT INTO audit_log (study_id, reviewer, decision, comment, ai_score, flagged,
               model_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (study_id, body.reviewer, body.decision, body.comment, study["score"],
             study["flagged"], study["model_version"], now))
        conn.execute("UPDATE studies SET review_status=? WHERE study_id=?", (body.decision, study_id))
    entry = conn.execute("SELECT * FROM audit_log WHERE id=?", (cur.lastrowid,)).fetchone()
    return {"review_status": body.decision, "audit_entry": dict(entry)}


@app.get("/audit")
def audit_log(study_id: Optional[str] = None, limit: int = Query(200, ge=1, le=1000),
              conn=Depends(get_conn)):
    sql, args = "SELECT * FROM audit_log", []
    if study_id:
        sql += " WHERE study_id=?"; args.append(study_id)
    rows = conn.execute(sql + " ORDER BY id DESC LIMIT ?", args + [limit]).fetchall()
    return {"count": len(rows), "entries": [dict(r) for r in rows]}


@app.get("/stats")
def stats(conn=Depends(get_conn)):
    q = lambda sql: conn.execute(sql).fetchone()[0]
    return {
        "total": q("SELECT COUNT(*) FROM studies"),
        "analysed": q("SELECT COUNT(*) FROM studies WHERE status='done'"),
        "flagged": q("SELECT COUNT(*) FROM studies WHERE flagged=1"),
        "pending_review": q("SELECT COUNT(*) FROM studies WHERE status='done' AND review_status='pending'"),
        "approved": q("SELECT COUNT(*) FROM studies WHERE review_status='approved'"),
        "rejected": q("SELECT COUNT(*) FROM studies WHERE review_status='rejected'"),
        "audit_entries": q("SELECT COUNT(*) FROM audit_log"),
    }