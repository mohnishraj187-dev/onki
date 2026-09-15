import os
import io
import re
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage import measure
import trimesh

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from segmentation_service import TumorSegmentationService


# ============================================================
# CONFIGURATION
# ============================================================

app = FastAPI(
    title="OncoMorph4D Backend",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MASK_DIR = DATA_DIR / "masks"
MODEL_DIR = DATA_DIR / "models"

CHECKPOINT_PATH = (
    BASE_DIR
    / "models"
    / "epoch_5_model.pth"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SCAN STORAGE
# ============================================================

scans = {}


# ============================================================
# TUMOUR SEGMENTATION MODEL
# ============================================================

segmentation_service = None


@app.on_event("startup")
def startup():

    global segmentation_service

    print()
    print("=" * 70)
    print("ONCOMORPH4D BACKEND STARTING")
    print("=" * 70)

    print(
        f"Checkpoint:\n"
        f"{CHECKPOINT_PATH}"
    )

    if not CHECKPOINT_PATH.exists():

        print()
        print(
            "WARNING: Tumour model checkpoint "
            "was not found."
        )

        print(
            "Place epoch_5_model.pth at:"
        )

        print(
            CHECKPOINT_PATH
        )

        print(
            "Tumour segmentation will not work."
        )

    else:

        try:

            segmentation_service = (
                TumorSegmentationService(
                    str(CHECKPOINT_PATH)
                )
            )

            print(
                "Tumour segmentation model loaded successfully."
            )

        except Exception as e:

            print()
            print(
                "ERROR LOADING TUMOUR MODEL:"
            )

            print(
                repr(e)
            )

            segmentation_service = None

    print("=" * 70)
    print()


# ============================================================
# HELPERS
# ============================================================

def safe_id(value):

    value = str(value).strip()

    value = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        value
    )

    if not value:

        raise HTTPException(
            status_code=400,
            detail="Invalid scan ID."
        )

    return value


def validate_nifti(filename):

    if not filename:
        return False

    filename = filename.lower()

    return (
        filename.endswith(".nii")
        or filename.endswith(".nii.gz")
    )


def save_upload(upload_file, destination):

    with open(
        destination,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            upload_file.file,
            buffer
        )


# ============================================================
# MODALITY DETECTION
# ============================================================

def detect_modality(filename):

    name = filename.lower()

    if (
        "_brain_t1n" in name
        or "_t1n" in name
        or "t1n" in name
    ):
        return "T1n"

    if (
        "_brain_t1c" in name
        or "_t1c" in name
        or "t1c" in name
    ):
        return "T1c"

    if (
        "_brain_t2w" in name
        or "_t2w" in name
        or "t2w" in name
    ):
        return "T2w"

    if (
        "_brain_t2f" in name
        or "_t2f" in name
        or "t2f" in name
        or "flair" in name
    ):
        return "T2f"

    return None


# ============================================================
# BRAIN GLB
# ============================================================

def generate_brain_glb(
    volume,
    output_path
):

    print()
    print(
        f"Generating brain GLB: "
        f"{output_path.name}"
    )

    data = np.asarray(
        volume,
        dtype=np.float32
    )

    data = np.nan_to_num(
        data
    )

    # --------------------------------------------------------
    # Smooth
    # --------------------------------------------------------

    smoothed = gaussian_filter(
        data,
        sigma=1.0
    )

    maximum = float(
        np.max(smoothed)
    )

    if maximum <= 0:

        raise ValueError(
            "MRI contains no usable intensity."
        )

    threshold = maximum * 0.15

    print(
        f"Maximum intensity: "
        f"{maximum:.3f}"
    )

    print(
        f"Marching cubes threshold: "
        f"{threshold:.3f}"
    )

    # --------------------------------------------------------
    # Marching cubes
    # --------------------------------------------------------

    vertices, faces, normals, values = (
        measure.marching_cubes(
            smoothed,
            level=threshold
        )
    )

    print(
        f"Generated mesh: "
        f"{len(vertices)} vertices, "
        f"{len(faces)} faces"
    )

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False
    )

    mesh.remove_unreferenced_vertices()

    # Remove invalid faces
    if len(mesh.faces) > 0:

        try:

            areas = mesh.area_faces

            valid = (
                np.isfinite(areas)
                & (areas > 1e-8)
            )

            mesh.update_faces(
                valid
            )

        except Exception as e:

            print(
                f"Face cleanup warning: {e}"
            )

    mesh.remove_unreferenced_vertices()

    # --------------------------------------------------------
    # Simplification
    # --------------------------------------------------------

    target_faces = 100000

    if len(mesh.faces) > target_faces:

        print(
            f"Simplifying mesh from "
            f"{len(mesh.faces)} faces..."
        )

        try:

            ratio = (
                target_faces
                / len(mesh.faces)
            )

            mesh = (
                mesh.simplify_quadric_decimation(
                    percent=ratio
                )
            )

            print(
                f"Simplified to "
                f"{len(mesh.faces)} faces"
            )

        except Exception as e:

            print(
                f"Mesh simplification skipped: {e}"
            )

    # --------------------------------------------------------
    # Center model
    # --------------------------------------------------------

    if len(mesh.vertices) > 0:

        try:
            brain_center = np.array(
                mesh.center_mass,
                dtype=np.float64
            )

        except Exception:

            brain_center = np.array(
                mesh.vertices.mean(axis=0),
                dtype=np.float64
            )

    else:

        raise ValueError(
            "Brain mesh contains no vertices."
        )

    mesh.apply_translation(
        -brain_center
    )
    # --------------------------------------------------------
    # Orientation
    # --------------------------------------------------------

    rotation = (
        trimesh.transformations.rotation_matrix(
            np.pi,
            [1, 0, 0]
        )
    )

    mesh.apply_transform(
        rotation
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    mesh.export(
        str(output_path),
        file_type="glb"
    )

    print(
        f"Brain GLB created successfully:"
        f"\n{output_path}"
    )

    return {
    "path": output_path,
    "center": brain_center
}


# ============================================================
# TUMOUR GLB
# ============================================================

def generate_tumor_glb(
    mask,
    output_path,
    brain_center
):

    print()
    print(
        f"Generating tumour GLB: "
        f"{output_path.name}"
    )

    binary = (
        np.asarray(mask) > 0
    ).astype(np.uint8)

    voxel_count = int(
        np.sum(binary)
    )

    print(
        f"Tumour voxels: "
        f"{voxel_count}"
    )

    if voxel_count == 0:

        raise ValueError(
            "Model produced an empty tumour mask."
        )

    # --------------------------------------------------------
    # Smooth mask
    # --------------------------------------------------------

    smoothed = gaussian_filter(
        binary.astype(np.float32),
        sigma=0.5
    )

    vertices, faces, normals, values = (
        measure.marching_cubes(
            smoothed,
            level=0.5
        )
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False
    )

    mesh.remove_unreferenced_vertices()

    # --------------------------------------------------------
    # Remove invalid faces
    # --------------------------------------------------------

    if len(mesh.faces) > 0:

        try:

            areas = mesh.area_faces

            valid = (
                np.isfinite(areas)
                & (areas > 1e-8)
            )

            mesh.update_faces(
                valid
            )

        except Exception as e:

            print(
                f"Tumour face cleanup warning: {e}"
            )

    mesh.remove_unreferenced_vertices()

    # --------------------------------------------------------
    # Simplification
    # --------------------------------------------------------

    target_faces = 50000

    if len(mesh.faces) > target_faces:

        try:

            ratio = (
                target_faces
                / len(mesh.faces)
            )

            mesh = (
                mesh.simplify_quadric_decimation(
                    percent=ratio
                )
            )

        except Exception as e:

            print(
                f"Tumour simplification skipped: {e}"
            )

    brain_center

    # --------------------------------------------------------
    # IMPORTANT:
    # Brain and tumour must use exactly
    # the same coordinate transformation.
    # --------------------------------------------------------

    rotation = (
        trimesh.transformations.rotation_matrix(
            np.pi,
            [1, 0, 0]
        )
    )

    mesh.apply_transform(
        rotation
    )

    # --------------------------------------------------------
    # Red tumour material
    # --------------------------------------------------------

    mesh.visual.material = (
        trimesh.visual.material.PBRMaterial(
            baseColorFactor=[
                1.0,
                0.05,
                0.05,
                1.0
            ],
            metallicFactor=0.0,
            roughnessFactor=0.5
        )
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    mesh.export(
        str(output_path),
        file_type="glb"
    )

    print(
        f"Tumour GLB created successfully:"
        f"\n{output_path}"
    )

    return output_path


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "OncoMorph4D backend",
        "tumour_model_loaded":
            segmentation_service is not None
    }


# ============================================================
# UPLOAD FOUR MRI MODALITIES
# ============================================================

@app.post("/api/upload-mri")
async def upload_mri(

    files: list[UploadFile] = File(...),

    scan_id: str = Form(...)
):

    # --------------------------------------------------------
    # Scan ID
    # --------------------------------------------------------

    scan_id = safe_id(
        scan_id
    )

    # --------------------------------------------------------
    # EXACTLY FOUR FILES
    # --------------------------------------------------------

    if len(files) != 4:

        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly 4 MRI files are required: "
                "T1n, T1c, T2w and T2f."
            )
        )

    # --------------------------------------------------------
    # Detect modalities
    # --------------------------------------------------------

    modality_files = {}

    for file in files:

        if not validate_nifti(
            file.filename
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid MRI file: "
                    f"{file.filename}"
                )
            )

        modality = detect_modality(
            file.filename
        )

        if modality is None:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not determine modality "
                    f"from filename: "
                    f"{file.filename}. "
                    f"Expected T1n, T1c, T2w or T2f."
                )
            )

        if modality in modality_files:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Duplicate modality: "
                    f"{modality}"
                )
            )

        modality_files[modality] = file

    required = [
        "T1n",
        "T1c",
        "T2w",
        "T2f"
    ]

    missing = [
        modality
        for modality in required
        if modality not in modality_files
    ]

    if missing:

        raise HTTPException(
            status_code=400,
            detail=(
                "Missing MRI modalities: "
                + ", ".join(missing)
            )
        )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"NEW FOUR-MODALITY SCAN: "
        f"{scan_id}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    saved_paths = {}

    for modality in required:

        upload = modality_files[
            modality
        ]

        destination = (
            UPLOAD_DIR
            / f"{scan_id}_{modality}.nii.gz"
        )

        save_upload(
            upload,
            destination
        )

        saved_paths[
            modality
        ] = destination

        print(
            f"{modality}: "
            f"{upload.filename}"
        )

    # --------------------------------------------------------
    # Load T1n
    # --------------------------------------------------------

    try:

        t1n_nii = nib.load(
            str(saved_paths["T1n"])
        )

        t1n_volume = (
            t1n_nii.get_fdata(
                dtype=np.float32
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not load T1n MRI: "
                f"{str(e)}"
            )
        )

    if t1n_volume.ndim != 3:

        raise HTTPException(
            status_code=400,
            detail=(
                "MRI must be a 3D volume."
            )
        )

    dimensions = list(
        t1n_volume.shape
    )

    print(
        f"MRI dimensions: "
        f"X={dimensions[0]}, "
        f"Y={dimensions[1]}, "
        f"Z={dimensions[2]}"
    )

    # --------------------------------------------------------
    # Verify dimensions of all modalities
    # --------------------------------------------------------

    for modality in required:

        nii = nib.load(
            str(saved_paths[modality])
        )

        shape = nii.shape

        if tuple(shape) != tuple(
            t1n_volume.shape
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Dimension mismatch. "
                    f"T1n={t1n_volume.shape}, "
                    f"{modality}={shape}"
                )
            )

    # ========================================================
    # 1. BRAIN GLB
    # ========================================================

    brain_glb = (
        MODEL_DIR
        / f"{scan_id}_brain.glb"
    )

    try:

        brain_result = generate_brain_glb(
            t1n_volume,
            brain_glb
        )

        brain_center = brain_result["center"]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Brain GLB generation failed: "
                f"{str(e)}"
            )
        )

    # ========================================================
    # 2. TUMOUR SEGMENTATION
    # ========================================================

    if segmentation_service is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Tumour segmentation model "
                "is not loaded. "
                "Check backend/models/"
                "epoch_5_model.pth"
            )
        )

    mask_path = (
        MASK_DIR
        / f"{scan_id}_tumorMask.nii.gz"
    )

    try:

        segmentation_result = (
            segmentation_service.predict(
                t1n_path=str(
                    saved_paths["T1n"]
                ),
                t1c_path=str(
                    saved_paths["T1c"]
                ),
                t2w_path=str(
                    saved_paths["T2w"]
                ),
                t2f_path=str(
                    saved_paths["T2f"]
                ),
                output_path=str(
                    mask_path
                )
            )
        )

    except Exception as e:

        print()
        print(
            "SEGMENTATION ERROR:"
        )
        print(
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Tumour segmentation failed: "
                f"{str(e)}"
            )
        )

    predicted_mask = np.asarray(
        segmentation_result["mask"]
    )

    # --------------------------------------------------------
    # Validate mask shape
    # --------------------------------------------------------

    if predicted_mask.shape != t1n_volume.shape:

        raise HTTPException(
            status_code=500,
            detail=(
                "Generated tumour mask shape does "
                "not match the MRI shape. "
                f"MRI={t1n_volume.shape}, "
                f"Mask={predicted_mask.shape}"
            )
        )

    tumour_voxels = int(
        np.sum(
            predicted_mask > 0
        )
    )

    if tumour_voxels == 0:

        raise HTTPException(
            status_code=500,
            detail=(
                "The segmentation model produced "
                "an empty tumour mask."
            )
        )

    # ========================================================
    # 3. TUMOUR GLB
    # ========================================================

    tumor_glb = (
        MODEL_DIR
        / f"{scan_id}_tumor.glb"
    )

    try:

        generate_tumor_glb(
            predicted_mask,
            tumor_glb,
            brain_center
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Tumour GLB generation failed: "
                f"{str(e)}"
            )
        )

    # ========================================================
    # STORE EVERYTHING
    # ========================================================

    scans[scan_id] = {

        "dimensions": dimensions,

        "volume": t1n_volume,

        "mask": predicted_mask,

        "modalities": {
            modality: str(
                saved_paths[modality]
            )
            for modality in required
        },

        "brain_glb": str(
            brain_glb
        ),

        "tumor_glb": str(
            tumor_glb
        ),

        "mask_path": str(
            mask_path
        ),

        "tumour_voxels":
            tumour_voxels
    }

    print()
    print("=" * 70)
    print(
        f"SCAN {scan_id} READY"
    )

    print(
        f"Brain GLB: "
        f"{brain_glb}"
    )

    print(
        f"Tumour GLB: "
        f"{tumor_glb}"
    )

    print(
        f"Tumour mask: "
        f"{mask_path}"
    )

    print(
        f"Tumour voxels: "
        f"{tumour_voxels}"
    )

    print("=" * 70)
    print()

    # ========================================================
    # RETURN TO FRONTEND
    #
    # IMPORTANT:
    # dimensions is now an OBJECT,
    # not a raw list.
    # ========================================================

    return {

        "success": True,

        "scan_id": scan_id,

        "dimensions": {
            "x": dimensions[0],
            "y": dimensions[1],
            "z": dimensions[2]
        },

        "modalities": required,

        "brain_model_url":
            f"/api/model/brain/{scan_id}",

        "tumor_model_url":
            f"/api/model/tumor/{scan_id}",

        "mask_available": True,

        "tumour_voxels":
            tumour_voxels
    }


# ============================================================
# DIMENSIONS
# ============================================================

@app.get(
    "/api/dimensions/{scan_id}"
)
def get_dimensions(
    scan_id: str
):

    if scan_id not in scans:

        raise HTTPException(
            status_code=404,
            detail="Scan not found."
        )

    dimensions = scans[
        scan_id
    ]["dimensions"]

    return {

        "x": dimensions[0],
        "y": dimensions[1],
        "z": dimensions[2]
    }


# ============================================================
# MATPLOTLIB SLICE
# ============================================================

@app.get(
    "/api/slice/{scan_id}/{plane}/{index}"
)
def get_slice(

    scan_id: str,

    plane: str,

    index: int,

    show_mask: bool = True
):

    if scan_id not in scans:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Scan '{scan_id}' not found."
            )
        )

    volume = scans[
        scan_id
    ]["volume"]

    mask = scans[
        scan_id
    ]["mask"]

    x_dim = volume.shape[0]
    y_dim = volume.shape[1]
    z_dim = volume.shape[2]

    # --------------------------------------------------------
    # Axial
    # --------------------------------------------------------

    if plane == "axial":

        if index < 0 or index >= z_dim:

            raise HTTPException(
                status_code=400,
                detail="Axial index out of range."
            )

        image = volume[
            :,
            :,
            index
        ]

        mask_slice = mask[
            :,
            :,
            index
        ]

    # --------------------------------------------------------
    # Coronal
    # --------------------------------------------------------

    elif plane == "coronal":

        if index < 0 or index >= y_dim:

            raise HTTPException(
                status_code=400,
                detail="Coronal index out of range."
            )

        image = volume[
            :,
            index,
            :
        ]

        mask_slice = mask[
            :,
            index,
            :
        ]

    # --------------------------------------------------------
    # Sagittal
    # --------------------------------------------------------

    elif plane == "sagittal":

        if index < 0 or index >= x_dim:

            raise HTTPException(
                status_code=400,
                detail="Sagittal index out of range."
            )

        image = volume[
            index,
            :,
            :
        ]

        mask_slice = mask[
            index,
            :,
            :
        ]

    else:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid plane. "
                "Use axial, coronal or sagittal."
            )
        )

    # --------------------------------------------------------
    # Rotate
    # --------------------------------------------------------

    image = np.rot90(
        image
    )

    mask_slice = np.rot90(
        mask_slice
    )

    # --------------------------------------------------------
    # Matplotlib
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(5, 5),
        dpi=120
    )

    fig.patch.set_facecolor(
        "black"
    )

    ax.set_facecolor(
        "black"
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    if show_mask and np.any(mask_slice > 0):

        # Create a binary tumour mask
        tumour_overlay = np.ma.masked_where(
            mask_slice <= 0,
            np.ones_like(mask_slice, dtype=np.float32)
        )

    # Overlay tumour in red
        ax.imshow(
            tumour_overlay,
            cmap="Reds",
            alpha=0.75,
            vmin=0,
            vmax=1
        )

    ax.axis(
        "off"
    )

    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=120,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="black"
    )

    plt.close(
        fig
    )

    buffer.seek(
        0
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )


# ============================================================
# BRAIN GLB
# ============================================================

@app.get(
    "/api/model/brain/{scan_id}"
)
def get_brain_model(
    scan_id: str
):

    if scan_id not in scans:

        raise HTTPException(
            status_code=404,
            detail="Scan not found."
        )

    path = scans[
        scan_id
    ]["brain_glb"]

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Brain GLB not found."
        )

    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=(
            f"{scan_id}_brain.glb"
        )
    )


# ============================================================
# TUMOUR GLB
# ============================================================

@app.get(
    "/api/model/tumor/{scan_id}"
)
def get_tumor_model(
    scan_id: str
):

    if scan_id not in scans:

        raise HTTPException(
            status_code=404,
            detail="Scan not found."
        )

    path = scans[
        scan_id
    ]["tumor_glb"]

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Tumour GLB not found."
        )

    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=(
            f"{scan_id}_tumor.glb"
        )
    )


# ============================================================
# MASK INFO
# ============================================================

@app.get(
    "/api/mask/{scan_id}"
)
def get_mask_info(
    scan_id: str
):

    if scan_id not in scans:

        raise HTTPException(
            status_code=404,
            detail="Scan not found."
        )

    return {

        "scan_id": scan_id,

        "mask_available": True,

        "tumour_voxels":
            scans[scan_id][
                "tumour_voxels"
            ],

        "mask_path":
            scans[scan_id][
                "mask_path"
            ]
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/api/health"
)
def health():

    return {

        "status": "ok",

        "tumour_model_loaded":
            segmentation_service is not None,

        "loaded_scans":
            list(scans.keys())
    }


# ============================================================
# RUN
# ============================================================
#
# From backend directory:
#
# python -m uvicorn main:app --reload --port 8000
#
# ============================================================