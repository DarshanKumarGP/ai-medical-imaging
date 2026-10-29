from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pydicom

files = sorted(Path("data/raw/dicom").glob("*.dcm"))
labels = pd.read_csv("data/raw/stage_2_train_labels.csv")

ds = pydicom.dcmread(files[0])
print("File:", files[0].name)
print("Modality:", ds.Modality)
print("Body part:", ds.get("BodyPartExamined"))
print("View position:", ds.get("ViewPosition"))
print("Patient sex:", ds.get("PatientSex"))
print("Patient age:", ds.get("PatientAge"))
print("Rows x Cols:", ds.Rows, "x", ds.Columns)
print("Pixel spacing:", ds.get("PixelSpacing"))
print("Photometric:", ds.PhotometricInterpretation)

img = ds.pixel_array
print("Pixel array shape:", img.shape, "| dtype:", img.dtype,
      "| min/max:", img.min(), img.max())

# Show the scan with its pneumonia boxes, if any
pid = files[0].stem
boxes = labels[(labels.patientId == pid) & (labels.Target == 1)]
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img, cmap="gray")
for _, r in boxes.iterrows():
    ax.add_patch(patches.Rectangle((r.x, r.y), r.width, r.height,
                                   fill=False, edgecolor="red", linewidth=2))
ax.set_title(f"{pid[:8]}...  boxes: {len(boxes)}")
ax.axis("off")
plt.show()