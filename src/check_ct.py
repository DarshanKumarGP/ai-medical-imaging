"""Check the DICOM path against the original NIfTI files: same grid, same image, and a Dice score
between the AI mask and the ground-truth label (expected: close to the Colab test results)."""
from pathlib import Path

import numpy as np
from monai.transforms import (Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
                              Orientationd, ScaleIntensityRanged, Spacingd)

from ct_model import HU_MAX, HU_MIN, PIXDIM
from db import VOLUMES_DIR, connect

NIFTI = Path("demo_files/ct_nifti")
KEYS = ["image", "label"]
reference_pipeline = Compose([
    LoadImaged(keys=KEYS), EnsureChannelFirstd(keys=KEYS), Orientationd(keys=KEYS, axcodes="RAS"),
    Spacingd(keys=KEYS, pixdim=PIXDIM, mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=KEYS, source_key="image", allow_smaller=True),
])

conn = connect()
rows = conn.execute("SELECT patient_id, volume_path, spleen_volume_ml FROM studies "
                    "WHERE modality='CT' AND volume_path IS NOT NULL").fetchall()
if not rows:
    print("No analysed CT studies in the database yet.")
for r in rows:
    case = r["patient_id"]
    img, lab = NIFTI / f"{case}.nii.gz", NIFTI / f"label_{case}.nii.gz"
    if not (img.exists() and lab.exists()):
        print(f"{case}: no NIfTI reference found in {NIFTI}, skipped")
        continue
    ref = reference_pipeline({"image": str(img), "label": str(lab)})
    ref_ct = ref["image"][0].numpy()
    ref_mask = ref["label"][0].numpy() == 1
    ct = np.load(VOLUMES_DIR / f"{r['volume_path']}_ct.npy")
    mask = np.load(VOLUMES_DIR / f"{r['volume_path']}_mask.npy").astype(bool)
    print(f"\n{case}")
    print("  grid shape  DICOM path:", ct.shape, "| NIfTI reference:", ref_ct.shape)
    if ct.shape != ref_ct.shape:
        print("  MISMATCH: the two paths do not use the same grid, so Dice cannot be compared.")
        continue
    diff = float(np.abs(ct.astype(np.float32) / 255 - ref_ct).max())
    dice = 2 * (mask & ref_mask).sum() / (mask.sum() + ref_mask.sum() + 1e-8)
    ref_ml = ref_mask.sum() * PIXDIM[0] * PIXDIM[1] * PIXDIM[2] / 1000
    print(f"  max image difference: {diff:.4f}  (should be below 0.01)")
    print(f"  Dice vs ground truth: {dice:.4f}")
    print(f"  spleen volume  AI: {r['spleen_volume_ml']:.0f} mL | ground truth: {ref_ml:.0f} mL")