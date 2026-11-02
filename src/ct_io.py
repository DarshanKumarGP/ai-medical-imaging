"""Turn a CT series (a list of DICOM slices) into one 3D volume with a world affine.

Pure NumPy, so it is easy to test. Raises Skip for studies that cannot be segmented by design.
"""
import numpy as np

MIN_SLICES = 30


class Skip(Exception):
    """The study cannot be analysed by design (this is not a crash)."""


def assemble_volume(slices):
    """slices: list of dicts with keys
         pix (rows, cols array), ipp (3,), iop (6,), spacing (row_spacing, col_spacing),
         slope, intercept
    Returns (volume_hu float32 of shape (cols, rows, n_slices), affine 4x4 in RAS world coordinates).
    """
    if len(slices) < MIN_SLICES:
        raise Skip(f"CT series has only {len(slices)} slice(s); 3D segmentation needs at least {MIN_SLICES}")
    if len({s["pix"].shape for s in slices}) != 1:
        raise Skip("slices of the series have different sizes")

    iop = np.asarray(slices[0]["iop"], dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]          # along a row (columns increase) / along a column (rows increase)
    normal = np.cross(row_dir, col_dir)
    if abs(normal[2]) < 0.95:
        raise Skip("series is not axial")

    pos = np.array([np.dot(s["ipp"], normal) for s in slices], dtype=float)
    order = np.argsort(pos)
    sl = [slices[i] for i in order]
    pos = pos[order]
    gaps = np.diff(pos)
    if gaps.min() <= 1e-3:
        raise Skip("duplicate slice positions")
    if np.abs(gaps - gaps.mean()).max() > 0.1 * gaps.mean():
        raise Skip("irregular slice spacing")

    ipp = np.array([s["ipp"] for s in sl], dtype=float)
    row_spacing, col_spacing = sl[0]["spacing"]
    step = (ipp[-1] - ipp[0]) / (len(sl) - 1)

    volume = np.stack(
        [(s["pix"].astype(np.float32) * float(s["slope"]) + float(s["intercept"])).T for s in sl], axis=-1
    )                                             # (cols, rows, slices)

    lps = np.eye(4)
    lps[:3, 0] = row_dir * col_spacing           # one step along i (columns)
    lps[:3, 1] = col_dir * row_spacing           # one step along j (rows)
    lps[:3, 2] = step                            # one step along k (slices)
    lps[:3, 3] = ipp[0]
    ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps  # DICOM is LPS, MONAI/NIfTI work in RAS
    return volume.astype(np.float32), ras