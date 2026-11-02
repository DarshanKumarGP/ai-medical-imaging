"""Render one axial slice of a stored CT volume (and its AI mask) as a PNG for the viewer."""
import io

import numpy as np
from PIL import Image

SIZE = 512
MASK_RGB = (239, 68, 68)
MASK_ALPHA = 130


def _orient(sl):
    """(x, y) slice in RAS -> image with anterior at the top and the patient's right on the left
    of the picture (radiological convention)."""
    return sl.T[::-1, ::-1]


def _square(arr, resample):
    h, w = arr.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side) + arr.shape[2:], dtype=arr.dtype)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = arr
    return Image.fromarray(canvas).resize((SIZE, SIZE), resample)


def _png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_scan(ct_slice):
    arr = np.ascontiguousarray(_orient(np.asarray(ct_slice, dtype=np.uint8)))
    return _png(_square(arr, Image.BILINEAR))


def render_mask(mask_slice):
    m = np.ascontiguousarray(_orient(np.asarray(mask_slice, dtype=np.uint8)))
    rgba = np.zeros(m.shape + (4,), dtype=np.uint8)
    rgba[m > 0] = (*MASK_RGB, MASK_ALPHA)
    return _png(_square(rgba, Image.NEAREST))