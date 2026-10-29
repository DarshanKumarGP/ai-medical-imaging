import subprocess
import zipfile
from pathlib import Path

import pandas as pd

COMP = "rsna-pneumonia-detection-challenge"
OUT = Path("data/raw/dicom")
NESTED = OUT / "stage_2_train_images"  # Kaggle keeps this subfolder name
OUT.mkdir(parents=True, exist_ok=True)

labels = pd.read_csv("data/raw/stage_2_train_labels.csv")

# One row per patient: 1 if any box exists, else 0
per_patient = labels.groupby("patientId")["Target"].max().reset_index()
pneumonia = per_patient[per_patient.Target == 1].sample(10, random_state=42)
normal = per_patient[per_patient.Target == 0].sample(10, random_state=42)
ids = list(pneumonia.patientId) + list(normal.patientId)

for pid in ids:
    target = OUT / f"{pid}.dcm"
    nested_file = NESTED / f"{pid}.dcm"

    # Already downloaded? Just make sure it sits in the right folder.
    if nested_file.exists() and not target.exists():
        nested_file.replace(target)
    if target.exists():
        continue

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMP,
         "-f", f"stage_2_train_images/{pid}.dcm", "-p", str(OUT)],
        check=True,
    )

    # Kaggle may save the file inside a subfolder: move it up one level
    if nested_file.exists():
        nested_file.replace(target)

    # Kaggle may also save it as a zip: extract it
    z = OUT / f"{pid}.dcm.zip"
    if z.exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(OUT)
        z.unlink()

# Remove the leftover subfolder if it is empty
if NESTED.exists() and not any(NESTED.iterdir()):
    NESTED.rmdir()

print("DICOM files downloaded:", len(list(OUT.glob("*.dcm"))))