import sqlite3
from pathlib import Path

import pandas as pd

from db import RUNTIME, connect

conn = connect()
df = pd.read_sql("SELECT * FROM studies", conn)
print("Status counts:\n", df.status.value_counts().to_string(), "\n")

done = df[df.status == "done"].copy()
truth = pd.read_csv("data/raw/stage_2_train_labels.csv").groupby("patientId").Target.max()
test_ids = set(pd.read_csv("data/splits/test.csv").patientId)
done["truth"] = done.patient_id.map(truth)
done["in_test"] = done.patient_id.isin(test_ids)

for name, d in [("all processed", done), ("held-out TEST patients only", done[done.in_test])]:
    tp = int(((d.flagged == 1) & (d.truth == 1)).sum()); fn = int(((d.flagged == 0) & (d.truth == 1)).sum())
    tn = int(((d.flagged == 0) & (d.truth == 0)).sum()); fp = int(((d.flagged == 1) & (d.truth == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    print(f"{name}: n={len(d)}  TP={tp} FN={fn} TN={tn} FP={fp}  sensitivity={sens:.2f}  specificity={spec:.2f}")

missing = [p for p in done.scan_path.dropna() if not (RUNTIME / p).exists()]
missing += [p for p in done.heatmap_path.dropna() if not (RUNTIME / p).exists()]
print("\nImages on disk: scans", int(done.scan_path.notna().sum()),
      "| heatmaps", int(done.heatmap_path.notna().sum()), "| missing files:", len(missing))

try:   # audit log must be append-only
    conn.execute("INSERT INTO audit_log(study_id, reviewer, decision, created_at) "
                 "VALUES ('test','test','approved','2026-01-01')")
    conn.execute("UPDATE audit_log SET decision='rejected'")
    print("AUDIT TEST FAILED: the log was editable")
except sqlite3.DatabaseError as e:
    print("Audit log is append-only:", e)
finally:
    conn.rollback()