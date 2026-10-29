import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from gradcam import THRESHOLD, load_model, predict_with_heatmap

COMP = "rsna-pneumonia-detection-challenge"
OUT = Path("data/raw/dicom_test")
NESTED = OUT / "stage_2_train_images"   # Kaggle keeps this subfolder name
OUT.mkdir(parents=True, exist_ok=True)

labels = pd.read_csv("data/raw/stage_2_train_labels.csv")
test = pd.read_csv("data/splits/test.csv")
ids = test[test.label == 1].sample(40, random_state=7).patientId.tolist()

for pid in ids:
    target = OUT / f"{pid}.dcm"
    if target.exists():
        continue
    subprocess.run(["kaggle", "competitions", "download", "-c", COMP,
                    "-f", f"stage_2_train_images/{pid}.dcm", "-p", str(OUT)], check=True)
    nested = NESTED / f"{pid}.dcm"
    if nested.exists():
        nested.replace(target)
if NESTED.exists() and not any(NESTED.iterdir()):
    NESTED.rmdir()

model = load_model()
K = 224 / 1024
rows = []
for pid in ids:
    score, _, heat = predict_with_heatmap(OUT / f"{pid}.dcm", model)
    boxes = labels[(labels.patientId == pid) & (labels.Target == 1)]
    mask = np.zeros((224, 224), dtype=bool)
    for _, r in boxes.iterrows():
        mask[int(r.y * K): int((r.y + r.height) * K) + 1,
             int(r.x * K): int((r.x + r.width) * K) + 1] = True
    py, px = np.unravel_index(heat.argmax(), heat.shape)
    rows.append(dict(patientId=pid, score=score, detected=bool(score >= THRESHOLD),
                     peak_in_box=bool(mask[py, px]), box_area=float(mask.mean()),
                     heat_in_box=float(heat[mask].sum() / heat.sum())))

df = pd.DataFrame(rows)
det = df[df.detected]
summary = dict(
    n_cases=len(df),
    detected=int(df.detected.sum()),
    peak_in_box_all=float(df.peak_in_box.mean()),
    peak_in_box_detected=float(det.peak_in_box.mean()) if len(det) else None,
    chance_level=float(df.box_area.mean()),
    heat_in_box_detected=float(det.heat_in_box.mean()) if len(det) else None,
)
df.to_csv("docs/gradcam_eval.csv", index=False)
json.dump(summary, open("docs/gradcam_eval.json", "w"), indent=2)
print(json.dumps(summary, indent=2))