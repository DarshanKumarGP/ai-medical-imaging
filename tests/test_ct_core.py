import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from PIL import Image

from ct_io import MIN_SLICES, Skip, assemble_volume
from ct_render import SIZE, render_mask, render_scan


def make_slices(ni=40, nj=30, nk=40, spacing=(0.82, 0.78, 5.0), seed=0):
    """Synthetic CT in canonical RAS+, stored the way src/nifti_to_dicom.py stores it."""
    rng = np.random.default_rng(seed)
    sx, sy, sz = spacing[1], spacing[0], spacing[2]
    aff = np.diag([sx, sy, sz, 1.0])
    aff[:3, 3] = [-190.3, -250.7, -120.0]
    data = rng.integers(-1000, 1000, size=(ni, nj, nk)).astype(np.int16)
    out = []
    for k in range(nk):
        pix = np.ascontiguousarray(data[::-1, ::-1, k].T)
        x_r, y_a, z = (aff @ np.array([ni - 1, nj - 1, k, 1.0]))[:3]
        out.append(dict(pix=pix, ipp=[-x_r, -y_a, z], iop=[1, 0, 0, 0, 1, 0],
                        spacing=[sy, sx], slope=1, intercept=0))
    return data, aff, out


def to_ras_plus(vol, affine):
    """Same reorientation MONAI's Orientationd(axcodes='RAS') performs."""
    v, a = vol.copy(), affine.copy()
    for ax in range(3):
        col = a[:3, ax]
        if col[np.argmax(np.abs(col))] < 0:
            v = np.flip(v, axis=ax)
            a[:3, 3] = a[:3, 3] + col * (v.shape[ax] - 1)
            a[:3, ax] = -col
    return v, a


def test_series_round_trip_matches_original_geometry():
    data, aff, slices = make_slices()
    vol, ras = assemble_volume(slices)
    v, a = to_ras_plus(vol, ras)
    assert np.array_equal(v, data)
    assert np.allclose(a, aff)


def test_slice_order_does_not_matter():
    _, _, slices = make_slices()
    vol, ras = assemble_volume(slices)
    shuffled = slices[::-1][::2] + slices[::-1][1::2]
    vol2, ras2 = assemble_volume(shuffled)
    assert np.array_equal(vol, vol2) and np.allclose(ras, ras2)


def test_rescale_slope_and_intercept_applied():
    _, _, slices = make_slices(nk=MIN_SLICES)
    for s in slices:
        s["slope"], s["intercept"] = 2, -1024
    vol, _ = assemble_volume(slices)
    raw = slices[0]["pix"].astype(np.float32).T
    # slices[0] is the lowest slice (z increases with k)
    assert np.array_equal(vol[..., 0], raw * 2 - 1024)


def test_single_slice_is_skipped():
    _, _, slices = make_slices(nk=MIN_SLICES)
    with pytest.raises(Skip, match="at least"):
        assemble_volume(slices[:1])


def test_irregular_spacing_is_skipped():
    _, _, slices = make_slices()
    slices[10]["ipp"][2] += 2.0
    with pytest.raises(Skip, match="irregular"):
        assemble_volume(slices)


def test_duplicate_positions_are_skipped():
    _, _, slices = make_slices()
    slices[5]["ipp"] = list(slices[4]["ipp"])
    with pytest.raises(Skip):
        assemble_volume(slices)


def test_non_axial_series_is_skipped():
    _, _, slices = make_slices()
    for s in slices:
        s["iop"] = [1, 0, 0, 0, 0, -1]    # coronal
    with pytest.raises(Skip, match="axial"):
        assemble_volume(slices)


def test_render_orientation_and_overlay():
    ct = np.zeros((60, 40), dtype=np.uint8)       # (x, y) slice on a 1.5 mm grid
    ct[50:55, 5:10] = 200                          # large x (patient right), small y (posterior)
    scan = Image.open(io.BytesIO(render_scan(ct)))
    assert scan.size == (SIZE, SIZE) and scan.mode == "L"
    a = np.asarray(scan)
    ys, xs = np.nonzero(a > 100)
    assert xs.mean() < SIZE / 2                    # patient's right shows on the LEFT of the picture
    assert ys.mean() > SIZE / 2                    # posterior shows at the BOTTOM

    mask = np.zeros((60, 40), dtype=np.uint8)
    mask[50:55, 5:10] = 1
    m = np.asarray(Image.open(io.BytesIO(render_mask(mask))))
    assert m.shape == (SIZE, SIZE, 4)
    assert m[..., 3].max() > 0 and (m[..., 3] == 0).mean() > 0.9   # mostly transparent
    ys2, xs2 = np.nonzero(m[..., 3] > 0)
    assert abs(xs2.mean() - xs.mean()) < 6 and abs(ys2.mean() - ys.mean()) < 6   # overlay aligns with scan