from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torchvision.transforms as T
from PIL import Image

SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet values (DenseNet was pretrained on these)


def load_dicom(path):
    """Read a DICOM file and return a uint8 grayscale array scaled 0-255."""
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    if ds.PhotometricInterpretation == "MONOCHROME1":  # inverted scans
        img = img.max() - img
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)


def to_pil(img):
    """Grayscale array -> 224x224 RGB image (DenseNet expects 3 channels)."""
    return Image.fromarray(img).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)


# Mild, realistic changes only. No horizontal flip: it would put the heart on the right side.
augment = T.Compose([
    T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ColorJitter(brightness=0.2, contrast=0.2),
])
to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

if __name__ == "__main__":
    files = sorted(Path("data/raw/dicom").glob("*.dcm"))[:4]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for i, f in enumerate(files):
        base = to_pil(load_dicom(f))
        axes[0, i].imshow(base)
        axes[0, i].set_title("resized 224x224")
        axes[1, i].imshow(augment(base))
        axes[1, i].set_title("augmented")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    Path("docs").mkdir(exist_ok=True)
    plt.savefig("docs/preprocessing_preview.png", dpi=120)

    tensor = to_tensor(base)
    print("Model input tensor shape:", tuple(tensor.shape))
    print("Tensor mean/std:", round(tensor.mean().item(), 3), round(tensor.std().item(), 3))
    plt.show()