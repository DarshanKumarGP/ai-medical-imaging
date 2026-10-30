import subprocess
from pathlib import Path

import pandas as pd

COMP = "rsna-pneumonia-detection-challenge"
OUT = Path("demo_files")
NESTED = OUT / "stage_2_train_images"   # Kaggle keeps this subfolder name
OUT.mkdir(exist_ok=True)

test = pd.read_csv("data/splits/test.csv")   # held-out patients: the model never trained on them
used = {p.stem for folder in ("data/raw/dicom", "data/raw/dicom_test")
        for p in Path(folder).glob("*.dcm")}
fresh = test[~test.patientId.isin(used)]      # not in Orthanc yet

pick = pd.concat([fresh[fresh.label == 1].sample(3, random_state=2026),
                  fresh[fresh.label == 0].sample(3, random_state=2026)])

for pid, label in zip(pick.patientId, pick.label):
    target = OUT / f"{pid}.dcm"
    if not target.exists():
        subprocess.run(["kaggle", "competitions", "download", "-c", COMP,
                        "-f", f"stage_2_train_images/{pid}.dcm", "-p", str(OUT)], check=True)
        nested = NESTED / f"{pid}.dcm"
        if nested.exists():
            nested.replace(target)
    print(f"{pid[:8]}  ground truth: {'PNEUMONIA' if label else 'normal'}")

if NESTED.exists() and not any(NESTED.iterdir()):
    NESTED.rmdir()
print("Files ready in", OUT.resolve())