from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from skimage import measure


def cleanup_mesh(vertices, faces, max_faces):
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    mesh.remove_unreferenced_vertices()

    if len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(
                face_count=max_faces
            )
        except Exception as exc:
            print(
                f"Mesh simplification skipped: {exc}"
            )

    mesh.remove_unreferenced_vertices()
    return mesh


def orient_mesh(mesh):
    """
    Same orientation used for both brain and tumour.
    """
    transform = np.eye(4)
    transform[1, 1] = -1
    transform[2, 2] = -1
    mesh.apply_transform(transform)
    return mesh


def create_brain_glb(
    mri_path: Path,
    output_path: Path,
):
    nii = nib.as_closest_canonical(
        nib.load(str(mri_path))
    )

    data = nii.get_fdata(dtype=np.float32)

    if data.ndim > 3:
        data = np.squeeze(data)

    smooth = gaussian_filter(
        data,
        sigma=1.0,
    )

    maximum = float(np.max(smooth))

    if maximum <= 0:
        raise ValueError(
            "MRI contains no positive intensity values."
        )

    threshold = maximum * 0.15

    print(f"Maximum intensity: {maximum:.3f}")
    print(
        f"Marching cubes threshold: "
        f"{threshold:.3f}"
    )

    vertices, faces, _, _ = measure.marching_cubes(
        smooth,
        level=threshold,
    )

    print(
        f"Generated mesh: "
        f"{len(vertices)} vertices, "
        f"{len(faces)} faces"
    )

    mesh = cleanup_mesh(
        vertices,
        faces,
        max_faces=120000,
    )

    # Put the brain at the renderer origin.
    brain_center = mesh.bounding_box.centroid
    mesh.apply_translation(-brain_center)

    orient_mesh(mesh)

    mesh.export(str(output_path))

    print(
        f"Brain GLB created successfully: "
        f"{output_path}"
    )

    return output_path


def get_brain_center(mri_path: Path):
    nii = nib.as_closest_canonical(
        nib.load(str(mri_path))
    )

    data = nii.get_fdata(dtype=np.float32)

    if data.ndim > 3:
        data = np.squeeze(data)

    smooth = gaussian_filter(
        data,
        sigma=1.0,
    )

    maximum = float(np.max(smooth))

    if maximum <= 0:
        raise ValueError(
            "MRI contains no positive intensity values."
        )

    vertices, faces, _, _ = measure.marching_cubes(
        smooth,
        level=maximum * 0.15,
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    return mesh.bounding_box.centroid


def create_tumor_glb(
    mask_path: Path,
    brain_mri_path: Path,
    output_path: Path,
):
    nii = nib.as_closest_canonical(
        nib.load(str(mask_path))
    )

    mask = nii.get_fdata(dtype=np.float32)

    if mask.ndim > 3:
        mask = np.squeeze(mask)

    binary = mask > 0.5

    if not np.any(binary):
        print(
            "Tumour mask is empty. "
            "Creating empty tumour GLB."
        )
        trimesh.Scene().export(str(output_path))
        return output_path

    smooth = gaussian_filter(
        binary.astype(np.float32),
        sigma=0.5,
    )

    vertices, faces, _, _ = measure.marching_cubes(
        smooth,
        level=0.5,
    )

    mesh = cleanup_mesh(
        vertices,
        faces,
        max_faces=60000,
    )

    # CRITICAL:
    # Do not center the tumour independently.
    # It must stay in the same coordinate system as the brain.
    brain_center = get_brain_center(
        brain_mri_path
    )

    mesh.apply_translation(-brain_center)
    orient_mesh(mesh)

    mesh.export(str(output_path))

    print(
        f"Tumour GLB created successfully: "
        f"{output_path}"
    )

    return output_path


def tumor_volume_cm3(mask_path: Path) -> float:
    nii = nib.as_closest_canonical(
        nib.load(str(mask_path))
    )

    mask = nii.get_fdata()

    if mask.ndim > 3:
        mask = np.squeeze(mask)

    zooms = nii.header.get_zooms()[:3]
    voxel_mm3 = float(
        zooms[0] * zooms[1] * zooms[2]
    )

    voxel_count = int(
        np.count_nonzero(mask > 0.5)
    )

    return (
        voxel_count * voxel_mm3 / 1000.0
    )
