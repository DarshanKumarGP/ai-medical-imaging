from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

OUT = Path("data/splits")
OUT.mkdir(parents=True, exist_ok=True)

labels = pd.read_csv("data/raw/stage_2_train_labels.csv")

# One row per unique patientId: 1 if any pneumonia box exists
patients = (labels.groupby("patientId")["Target"].max()
            .reset_index().rename(columns={"Target": "label"}))

# 70% train, 15% val, 15% test, keeping the pneumonia rate equal in each
train, temp = train_test_split(patients, test_size=0.30,
                               stratify=patients.label, random_state=42)
val, test = train_test_split(temp, test_size=0.50,
                             stratify=temp.label, random_state=42)

# Safety check: no patient may appear in two splits
assert not set(train.patientId) & set(val.patientId)
assert not set(train.patientId) & set(test.patientId)
assert not set(val.patientId) & set(test.patientId)

for name, df in [("train", train), ("val", val), ("test", test)]:
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:5s} patients: {len(df):6d} | pneumonia rate: {df.label.mean():.3f}")