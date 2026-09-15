from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np

# FastAPI processes requests in worker threads. Never use Tk/Qt here.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_nifti(path: Path):
    nii = nib.as_closest_canonical(nib.load(str(path)))
    data = nii.get_fdata(dtype=np.float32)

    if data.ndim > 3:
        data = np.squeeze(data)

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {data.shape}")

    return nii, data


def dimensions(data: np.ndarray) -> dict:
    x, y, z = map(int, data.shape)
    return {"x": x, "y": y, "z": z}


def normalize_volume(data: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(
        data.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    lo = float(np.min(data))
    hi = float(np.max(data))

    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)

    return ((data - lo) / (hi - lo)).astype(np.float32)


def extract_slice(
    data: np.ndarray,
    mask: np.ndarray | None,
    plane: str,
    index: int,
):
    x, y, z = data.shape
    sizes = {
        "sagittal": x,
        "coronal": y,
        "axial": z,
    }

    if plane not in sizes:
        raise ValueError("Plane must be axial, coronal, or sagittal.")

    index = max(0, min(int(index), sizes[plane] - 1))

    if plane == "axial":
        image = data[:, :, index]
        overlay = mask[:, :, index] if mask is not None else None

    elif plane == "coronal":
        image = data[:, index, :]
        overlay = mask[:, index, :] if mask is not None else None

    else:
        image = data[index, :, :]
        overlay = mask[index, :, :] if mask is not None else None

    return np.rot90(image), (
        np.rot90(overlay) if overlay is not None else None
    )


def render_slice_png(
    data: np.ndarray,
    mask: np.ndarray | None,
    plane: str,
    index: int,
) -> BytesIO:
    image, overlay = extract_slice(data, mask, plane, index)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    fig.patch.set_facecolor("#05080b")
    ax.set_facecolor("#05080b")

    finite = image[np.isfinite(image)]

    if finite.size:
        vmin, vmax = np.percentile(finite, [1, 99])
        if vmax <= vmin:
            vmin = float(finite.min())
            vmax = float(finite.max()) + 1e-6
    else:
        vmin, vmax = 0, 1

    ax.imshow(
        image,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    if overlay is not None and np.any(overlay > 0):
        rgba = np.zeros((*overlay.shape, 4), dtype=np.float32)
        rgba[..., 0] = 1.0
        rgba[..., 1] = 0.0
        rgba[..., 2] = 0.0
        rgba[..., 3] = (overlay > 0).astype(np.float32) * 0.52
        ax.imshow(rgba, interpolation="nearest")

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    output = BytesIO()
    fig.savefig(
        output,
        format="png",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    output.seek(0)
    return output
