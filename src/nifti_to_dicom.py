from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CT_STORAGE = "1.2.840.10008.5.1.4.1.1.2"   # CT Image Storage
SRC, DST = Path("demo_files/ct_nifti"), Path("demo_files/ct")
PYDICOM2 = int(pydicom.__version__.split(".")[0]) < 3


def save(ds, path):
    if PYDICOM2:
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(path, write_like_original=False)
    else:
        ds.save_as(path, enforce_file_format=True)


def convert(nifti_path):
    case = nifti_path.name.replace(".nii.gz", "")            # e.g. spleen_12
    img = nib.as_closest_canonical(nib.load(str(nifti_path)))  # RAS+ orientation
    data = np.rint(img.get_fdata(dtype=np.float32)).astype(np.int16)   # HU values (X, Y, Z)
    aff = img.affine
    sx, sy, sz = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
    ni, nj, nk = data.shape
    study_uid, series_uid, frame_uid = generate_uid(), generate_uid(), generate_uid()
    out = DST / case
    out.mkdir(parents=True, exist_ok=True)

    for k in range(nk):
        # DICOM axial convention: columns run towards the patient's left, rows towards posterior
        pixels = np.ascontiguousarray(data[::-1, ::-1, k].T)
        x_r, y_a, z = (aff @ np.array([ni - 1, nj - 1, k, 1.0]))[:3]
        sop_uid = generate_uid()

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CT_STORAGE
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()

        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = CT_STORAGE, sop_uid
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.Modality = "CT"
        ds.PatientID, ds.PatientName = case, case
        ds.StudyDate = "19010101"
        ds.StudyDescription = "Spleen CT (MSD Task09, education demo)"
        ds.SeriesDescription = "axial"
        ds.SeriesNumber, ds.InstanceNumber = 1, k + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [float(-x_r), float(-y_a), float(z)]
        ds.PixelSpacing = [float(sy), float(sx)]
        ds.SliceThickness, ds.SliceLocation = float(sz), float(z)
        ds.Rows, ds.Columns = nj, ni
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 1
        ds.RescaleIntercept, ds.RescaleSlope = 0, 1
        ds.WindowCenter, ds.WindowWidth = 40, 400
        ds.PixelData = pixels.astype("<i2").tobytes()
        save(ds, out / f"slice_{k:04d}.dcm")

    # round-trip check on the middle slice
    mid = nk // 2
    back = pydicom.dcmread(out / f"slice_{mid:04d}.dcm").pixel_array
    assert np.array_equal(back, np.ascontiguousarray(data[::-1, ::-1, mid].T)), "round-trip failed"
    size_mb = sum(p.stat().st_size for p in out.glob("*.dcm")) / 1e6
    print(f"{case}: {nk} slices, {ni}x{nj}, spacing {sx:.2f}x{sy:.2f}x{sz:.2f} mm, "
          f"HU {data.min()}..{data.max()}, {size_mb:.0f} MB -> {out}")


if __name__ == "__main__":
    files = [p for p in sorted(SRC.glob("*.nii.gz")) if not p.name.startswith("label_")]
    assert files, f"No CT files found in {SRC}"
    for f in files:
        convert(f)