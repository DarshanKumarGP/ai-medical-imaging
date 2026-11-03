"""Write pinned requirements files from the packages installed in YOUR working venv,
so the Docker images use exactly the versions you tested with.

Run from the project root with the venv active:  python docker/make_requirements.py
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

API = ["fastapi", "uvicorn", "numpy", "pillow"]
INFERENCE = ["numpy", "pillow", "pydicom", "requests", "matplotlib", "grad-cam", "monai", "nibabel"]
TORCH = ["torch", "torchvision"]


def pins(names, extras=None):
    lines, missing = [], []
    for name in names:
        try:
            v = version(name).split("+")[0]          # drop local tags like "+cpu"
        except PackageNotFoundError:
            missing.append(name)
            continue
        lines.append(f"{name}[{extras[name]}]=={v}" if extras and name in extras else f"{name}=={v}")
    return lines, missing


total_missing = []
for filename, names, extras in [
    ("requirements-api.txt", API, {"uvicorn": "standard"}),
    ("requirements-inference.txt", INFERENCE, None),
    ("requirements-torch.txt", TORCH, None),
]:
    lines, missing = pins(names, extras)
    total_missing += missing
    Path(filename).write_text("\n".join(lines) + "\n")
    print(f"{filename}:", ", ".join(lines))

if total_missing:
    raise SystemExit(f"\nNot installed in this environment: {total_missing}. Activate the venv and retry.")
print("\nDone. Commit these three files.")