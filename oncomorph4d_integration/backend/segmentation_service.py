"""Inference and quantitative measurements for the trained 4-channel glioma model."""
from __future__ import annotations

from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import label
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from config import MODEL_CHECKPOINT, MULTICLASS_MODEL_CHECKPOINT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: SegResNet | None = None
MULTICLASS_MODEL: SegResNet | None = None
# Calibrated on labeled MU-Glioma-Post validation cases. The default argmax
# threshold (0.50) was systematically too liberal at lesion boundaries.
TUMOR_PROBABILITY_THRESHOLD = 0.70


def load_multiclass_model() -> SegResNet:
    global MULTICLASS_MODEL
    if MULTICLASS_MODEL is None:
        model = SegResNet(spatial_dims=3, in_channels=4, out_channels=5, init_filters=16, dropout_prob=0.1).to(DEVICE)
        checkpoint = torch.load(MULTICLASS_MODEL_CHECKPOINT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        MULTICLASS_MODEL = model
    return MULTICLASS_MODEL


def load_model() -> SegResNet:
    """Load the checkpoint produced by train_binary_segmentation.py exactly."""
    global MODEL
    if MODEL is None:
        if not MODEL_CHECKPOINT.exists():
            raise FileNotFoundError(f"Segmentation checkpoint not found: {MODEL_CHECKPOINT}")
        model = SegResNet(spatial_dims=3, in_channels=4, out_channels=2, init_filters=16, dropout_prob=0.1).to(DEVICE)
        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        MODEL = model
    return MODEL


def _load_and_normalize(paths: list[Path]) -> tuple[list[nib.Nifti1Image], torch.Tensor]:
    images = [nib.load(str(path)) for path in paths]
    shapes = {image.shape for image in images}
    if len(shapes) != 1 or len(next(iter(shapes))) != 3:
        raise ValueError("T1c, T1n, T2-FLAIR, and T2w must be aligned 3D volumes with identical shapes.")
    channels = []
    for image in images:
        data = np.asarray(image.dataobj, dtype=np.float32)
        values = data[data != 0]
        channels.append((data - values.mean()) / max(values.std(), 1e-6) if values.size else data)
    return images, torch.from_numpy(np.stack(channels)[None]).to(DEVICE)


def predict_tumor(modality_paths: list[Path], output_path: Path) -> dict:
    """Produce a binary NIfTI segmentation from [T1c, T1n, T2-FLAIR, T2w]."""
    images, tensor = _load_and_normalize(modality_paths)
    use_multiclass = MULTICLASS_MODEL_CHECKPOINT.exists()
    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
        logits = sliding_window_inference(tensor, (96, 96, 96), 1,
                                          load_multiclass_model() if use_multiclass else load_model(), overlap=0.5)
        if use_multiclass:
            mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
        else:
            tumor_probability = torch.softmax(logits, dim=1)[:, 1]
            mask = (tumor_probability >= TUMOR_PROBABILITY_THRESHOLD)[0].cpu().numpy().astype(np.uint8)
    # Remove isolated false-positive islands. A glioma mask should be a
    # contiguous lesion (or a small number of substantial components), not
    # scattered single voxels across the brain.
    components, count = label(mask > 0) if not use_multiclass else (None, 0)
    if count:
        sizes = np.bincount(components.ravel())
        # For this single-lesion glioma checkpoint, retain only the dominant
        # connected lesion. Tiny islands are model noise, not a second tumor.
        largest = int(np.argmax(sizes[1:]) + 1)
        mask = (components == largest).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, images[0].affine, images[0].header), str(output_path))
    voxel_ml = float(np.prod(images[0].header.get_zooms()[:3]) / 1000.0)
    coords = np.argwhere(mask > 0)
    class_voxels = {str(i): int((mask == i).sum()) for i in range(1, 5)}
    coords = np.argwhere(mask > 0)
    bbox = np.concatenate((coords.min(axis=0), coords.max(axis=0))).tolist() if len(coords) else None
    return {"mask_path": str(output_path), "tumor_voxels": int((mask > 0).sum()),
            "tumor_volume_ml": round(float((mask > 0).sum() * voxel_ml), 3),
            "edema_volume_ml": round(class_voxels["2"] * voxel_ml, 3),
            "core_volume_ml": round((class_voxels["1"] + class_voxels["3"] + class_voxels["4"]) * voxel_ml, 3),
            "class_voxels": class_voxels,
            "model_type": "multiclass" if use_multiclass else "binary",
            "centroid_voxel": coords.mean(axis=0).round(1).tolist() if len(coords) else None,
            "mask_bbox_voxel": bbox,
            "mask_components_removed": max(0, count - 1),
            "tumor_probability_threshold": TUMOR_PROBABILITY_THRESHOLD,
            "confidence_note": "Research segmentation output; clinician review required."}
