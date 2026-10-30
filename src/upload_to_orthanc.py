import os
from pathlib import Path

import requests

URL = os.getenv("ORTHANC_URL", "http://localhost:8042")
AUTH = (os.getenv("ORTHANC_USER", "demo"), os.getenv("ORTHANC_PASSWORD", "demo"))


def upload_all(folder="data/raw"):
    """Send every .dcm file under `folder` to Orthanc."""
    files = sorted(Path(folder).rglob("*.dcm"))
    for f in files:
        r = requests.post(f"{URL}/instances", data=f.read_bytes(), auth=AUTH, timeout=60)
        r.raise_for_status()
    return len(files)


if __name__ == "__main__":
    n = upload_all()
    print("Uploaded", n, "DICOM files")
    studies = requests.get(f"{URL}/studies", auth=AUTH, timeout=30).json()
    print("Studies in Orthanc:", len(studies))
    info = requests.get(f"{URL}/studies/{studies[0]}", auth=AUTH, timeout=30).json()
    print("Example study:", info["MainDicomTags"])
    print("Example patient:", info["PatientMainDicomTags"])