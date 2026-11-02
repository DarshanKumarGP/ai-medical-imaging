"""3D U-Net spleen segmentation. Uses exactly the preprocessing and network from training
(notebooks/02_train_segmentation.ipynb), so results match the Colab numbers."""
from pathlib import Path

import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import CropForegroundd, Compose, Orientationd, ScaleIntensityRanged, Spacingd

CT_MODEL_PATH = Path("models/best_unet3d.pt")
CT_MODEL_VERSION = "monai-unet3d-spleen-v1"
PIXDIM = (1.5, 1.5, 2.0)          # mm, as in training
HU_MIN, HU_MAX = -57, 164         # soft-tissue window, as in training
ROI = (96, 96, 96)


def load_ct_model(path=CT_MODEL_PATH):
    net = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
               strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval()


def preprocess(volume_hu, affine_ras):
    """(cols, rows, slices) HU volume + RAS affine -> (1, X, Y, Z) tensor on the training grid."""
    img = MetaTensor(torch.from_numpy(np.ascontiguousarray(volume_hu, dtype=np.float32))[None],
                     affine=torch.as_tensor(affine_ras, dtype=torch.float64))
    tf = Compose([
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=PIXDIM, mode="bilinear"),
        ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys="image", source_key="image", allow_smaller=True),
    ])
    return tf({"image": img})["image"]


def segment(volume_hu, affine_ras, model):
    x = preprocess(volume_hu, affine_ras)
    inp = x.as_tensor().float().unsqueeze(0)                       # (1, 1, X, Y, Z)
    with torch.inference_mode():
        logits = sliding_window_inference(inp, ROI, 1, model)      # batch 1 keeps CPU memory low
        prob = torch.softmax(logits.float(), dim=1)[0, 1]
    mask_t = prob > 0.5
    mask = mask_t.numpy()
    n_voxels = int(mask.sum())
    voxel_ml = PIXDIM[0] * PIXDIM[1] * PIXDIM[2] / 1000.0
    ct_u8 = np.ascontiguousarray((inp[0, 0].clamp(0, 1) * 255).round().byte().numpy())
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
    best = int(np.argmax(mask_u8.sum(axis=(0, 1)))) if n_voxels else ct_u8.shape[2] // 2
    return dict(
        ct=ct_u8, mask=mask_u8, n_slices=int(ct_u8.shape[2]), best_slice=best,
        volume_ml=n_voxels * voxel_ml,
        confidence=float(prob[mask_t].mean()) if n_voxels else None,
    )