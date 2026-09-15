import os

import nibabel as nib
import numpy as np
import trimesh

from skimage import measure
from scipy.ndimage import gaussian_filter, label, binary_closing, binary_fill_holes


# =========================================================
# COMMON MESH CLEANUP
# =========================================================

def clean_mesh(mesh):

    try:
        mesh.update_faces(
            mesh.nondegenerate_faces()
        )
    except Exception:
        pass

    try:
        mesh.update_faces(
            mesh.unique_faces()
        )
    except Exception:
        pass

    mesh.process(
        validate=True
    )

    return mesh


def apply_material(mesh, color, roughness=0.55, metallic=0.0, alpha=255):
    """Embed a PBR material so viewers render the clinical layers consistently."""
    rgba = tuple(int(round(channel * 255)) if 0 <= channel <= 1 else int(channel) for channel in color) + (int(alpha),)
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        name="OncoMorph material",
        baseColorFactor=rgba,
        alphaMode="BLEND" if alpha < 255 else "OPAQUE",
        doubleSided=True,
        metallicFactor=metallic,
        roughnessFactor=roughness,
    )


def create_combined_glb(brain_path: str, tumor_path: str, output_path: str):
    """Package registered brain and tumor meshes into one scene.

    A single GLB is important: separate model-viewer elements independently
    frame their contents, which visually breaks registration even when their
    voxel coordinates are identical.
    """
    scene = trimesh.Scene()
    brain = trimesh.load(brain_path, force="mesh")
    tumor = trimesh.load(tumor_path, force="mesh")
    scene.add_geometry(brain, node_name="brain")
    scene.add_geometry(tumor, node_name="tumor")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.export(output_path)
    return output_path


# =========================================================
# MRI → BRAIN GLB
# =========================================================

def create_brain_glb(
    nifti_path: str,
    output_path: str
):

    print("Creating brain GLB...")

    nii = nib.load(nifti_path)

    # Preserve the dataset-native voxel frame used by the trained model and
    # the 2D viewer. Brain and tumor are generated from the same frame.

    data = nii.get_fdata(
        dtype=np.float32
    )

    # Same basic idea as your extract.py:
    # smooth MRI before extracting surface
    data = gaussian_filter(data, sigma=1.8)

    maximum = np.max(data)

    if maximum <= 0:

        raise ValueError(
            "MRI volume contains no positive intensity."
        )

    # A higher isosurface level follows the brain boundary more closely and
    # avoids the hazy low-intensity halo that makes the surface look amorphous.
    threshold = max(maximum * 0.30, np.percentile(data[data > 0], 50))

    # Keep the dominant connected tissue region when low-intensity noise creates
    # small detached islands in the volume.
    tissue = data >= threshold
    # Build a clean outer envelope rather than exposing every voxel-level
    # intensity ridge in the live 3D viewer.
    tissue = binary_closing(tissue, iterations=3)
    tissue = binary_fill_holes(tissue)
    labels, count = label(tissue)
    if count:
        sizes = np.bincount(labels.ravel())
        tissue = labels == (np.argmax(sizes[1:]) + 1)
        data = data * tissue

    render_step = 2
    render_tissue = tissue[::render_step, ::render_step, ::render_step].astype(np.float32)
    verts, faces, normals, _ = measure.marching_cubes(render_tissue, level=0.5)
    verts *= render_step

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals
    )

    mesh = clean_mesh(mesh)
    # Smooth the extracted outer shell so the browser viewer presents an
    # anatomical envelope instead of a stair-stepped voxel surface.
    try:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.45, nu=0.53, iterations=12)
        mesh.fix_normals()
    except Exception as exc:
        print(f"Brain surface smoothing skipped: {exc}")
    # Warm skin-tone shell for easier anatomical interpretation.
    apply_material(mesh, (0.82, 0.46, 0.32), roughness=0.5, metallic=0.0, alpha=120)

    # Keep browser model reasonably light
    target_faces = 100000

    if len(mesh.faces) > target_faces:

        try:

            mesh = mesh.simplify_quadric_decimation(
                percent=(
                    target_faces
                    /
                    len(mesh.faces)
                )
            )

        except Exception as exc:

            print(
                f"Mesh simplification skipped: {exc}"
            )

    # Keep canonical voxel coordinates intact. The 2D viewer and tumor mesh
    # use the same canonical NIfTI orientation; applying an additional 180°
    # rotation here would make superior/inferior locations disagree.

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    mesh.export(
        output_path
    )

    print(
        f"Brain GLB saved: {output_path}"
    )

    return output_path


# =========================================================
# TUMOUR MASK → GLB
# =========================================================

def create_tumor_glb(
    mask_path: str,
    output_path: str
):

    print("Creating tumour GLB...")

    nii = nib.load(mask_path)

    # Preserve the dataset-native voxel frame used by the trained model and
    # the 2D viewer.

    mask = nii.get_fdata(
        dtype=np.float32
    )

    # Binary mask
    mask = (
        mask > 0.5
    ).astype(np.float32)

    if not np.any(mask):

        raise ValueError(
            "Tumour segmentation produced an empty mask."
        )

    # Slight smoothing gives marching cubes
    # a cleaner surface.
    smooth_mask = gaussian_filter(
        mask,
        sigma=0.5
    )

    verts, faces, normals, _ = (
        measure.marching_cubes(
            smooth_mask,
            level=0.5
        )
    )

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals
    )

    mesh = clean_mesh(mesh)
    # Tumour is intentionally high-contrast for quick visual localization.
    apply_material(mesh, (0.92, 0.035, 0.045), roughness=0.28, metallic=0.0)

    # Tumour model can be smaller
    target_faces = 50000

    if len(mesh.faces) > target_faces:

        try:

            mesh = mesh.simplify_quadric_decimation(
                percent=(
                    target_faces
                    /
                    len(mesh.faces)
                )
            )

        except Exception as exc:

            print(
                f"Tumour simplification skipped: {exc}"
            )

    # IMPORTANT:
    # Do NOT center the tumour independently.
    #
    # We need tumour coordinates to remain in the
    # SAME coordinate system as the brain.
    #
    # Therefore we apply the same orientation transform
    # but do not subtract its own center of mass.

    # No extra orientation transform: this must remain aligned with the
    # canonical mask used by the 2D viewer and combined GLB.

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    mesh.export(
        output_path
    )

    print(
        f"Tumour GLB saved: {output_path}"
    )

    return output_path


def create_multiclass_tumor_glb(mask_path: str, output_path: str):
    """Export one registered mesh per mask class with distinct clinical colors."""
    nii = nib.load(mask_path)
    mask = nii.get_fdata(dtype=np.float32).astype(np.uint8)
    # For the registered 3D overview, retain the dominant connected abnormal
    # region so scattered single-voxel islands do not obscure the lesion.
    connected, count = label(mask > 0)
    if count:
        sizes = np.bincount(connected.ravel())
        largest = np.argmax(sizes[1:]) + 1
        mask = np.where(connected == largest, mask, 0).astype(np.uint8)
    scene = trimesh.Scene()
    colors = {1: (123, 97, 255), 2: (242, 201, 76), 3: (242, 153, 74), 4: (235, 87, 87)}
    names = {1: "necrotic/non-enhancing", 2: "edema", 3: "non-enhancing-core", 4: "enhancing-tumor"}
    for value, color in colors.items():
        raw = (mask == value).astype(np.uint8)
        labels, count = label(raw)
        if count:
            sizes = np.bincount(labels.ravel())
            # Keep clinically meaningful connected regions; tiny isolated
            # specks are excluded from visualization only, not measurements.
            keep = np.where(sizes >= 100)[0]
            keep = keep[keep != 0]
            raw = np.isin(labels, keep).astype(np.float32)
        binary = gaussian_filter(raw, sigma=0.5)
        if binary.max() < 0.5:
            continue
        verts, faces, normals, _ = measure.marching_cubes(binary, level=0.5)
        mesh = clean_mesh(trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals))
        apply_material(mesh, tuple(channel / 255 for channel in color), roughness=0.32)
        scene.add_geometry(mesh, node_name=names[value])
    if not scene.geometry:
        raise ValueError("Multiclass mask contains no renderable classes.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.export(output_path)
    return output_path
