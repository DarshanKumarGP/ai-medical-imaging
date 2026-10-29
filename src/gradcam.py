from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torchvision import models

from preprocess import load_dicom, to_pil, to_tensor

MODEL_PATH = Path("models/best_model.pt")
THRESHOLD = 0.4837            # chosen on the validation set (see docs/metrics.json)
MODEL_VERSION = "densenet121-rsna-v1"


class CAMWrapper(nn.Module):
    """Same forward pass as torchvision's DenseNet, but with a normal (not in-place)
    ReLU, because in-place ops break Grad-CAM's hooks."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        f = F.relu(self.net.features(x))
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.net.classifier(f)


def load_model(path=MODEL_PATH):
    net = models.densenet121(weights=None)
    net.classifier = nn.Linear(net.classifier.in_features, 1)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return CAMWrapper(net).eval()


def predict_with_heatmap(dicom_source, model):
    """DICOM file (path or file-like) -> (score, overlay_image, raw_heatmap)."""
    pil = to_pil(load_dicom(dicom_source))
    x = to_tensor(pil).unsqueeze(0)

    # Grad-CAM on the last DenseNet block (norm5), explaining the "pneumonia" output
    with GradCAM(model=model, target_layers=[model.net.features[-1]]) as cam:
        heat = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0]

    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()

    rgb = np.asarray(pil, dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)
    return score, overlay, heat


if __name__ == "__main__":
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import pandas as pd

    labels = pd.read_csv("data/raw/stage_2_train_labels.csv")
    split_of = {}
    for s in ["train", "val", "test"]:
        for pid in pd.read_csv(f"data/splits/{s}.csv").patientId:
            split_of[pid] = s

    model = load_model()
    files = sorted(Path("data/raw/dicom").glob("*.dcm"))

    fig, axes = plt.subplots(4, 5, figsize=(18, 16))
    for ax, f in zip(axes.ravel(), files):
        score, overlay, _ = predict_with_heatmap(f, model)
        flagged = score >= THRESHOLD
        # Heatmap only for flagged scans; plain scan otherwise
        ax.imshow(overlay if flagged else to_pil(load_dicom(f)))
        boxes = labels[(labels.patientId == f.stem) & (labels.Target == 1)]
        k = 224 / 1024
        for _, r in boxes.iterrows():
            ax.add_patch(patches.Rectangle((r.x * k, r.y * k), r.width * k, r.height * k,
                                           fill=False, edgecolor="cyan", linewidth=1.5))
        truth = "PNEUMONIA" if len(boxes) else "normal"
        pred = "PNEUMONIA" if flagged else "normal"
        ax.set_title(f"true: {truth} | pred: {pred}\nscore {score:.2f} [{split_of[f.stem]}]",
                     fontsize=9, color="green" if truth == pred else "red")
        ax.axis("off")
    plt.tight_layout(h_pad=3)
    plt.savefig("docs/gradcam_samples.png", dpi=110)
    print("Saved docs/gradcam_samples.png")