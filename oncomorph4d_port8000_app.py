from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from skimage import measure
import trimesh

from pathlib import Path
import io
import os
import shutil
import json
import html
import zipfile

from segmentation_service import predict_tumor
from mesh_service import create_brain_glb, create_tumor_glb, create_multiclass_tumor_glb, create_combined_glb
from config import MASK_DIR


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(title="OncoMorph 4D Backend")

REACT_UI_DIR = Path("/home/mohnish/brain/frontend-brain2/dist")
if (REACT_UI_DIR / "assets").exists():
    app.mount("/brain2/assets", StaticFiles(directory=str(REACT_UI_DIR / "assets")), name="brain2-assets")

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

RAW_DATA_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"
TREND_FILE = DATA_DIR / "trend_history.json"
DATASET_ROOT = Path("/home/mohnish/Downloads/PKG - MU-Glioma-Post/MU-Glioma-Post")

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"


# ============================================================
# IN-MEMORY SCAN STORAGE
# ============================================================

# Each uploaded MRI is stored like:
#
# scans = {
#     "T1": {
#         "volume": numpy_array,
#         "dimensions": [x, y, z],
#         "filename": "...",
#         "model_path": "...",
#     }
# }

scans = {}
trend_history = json.loads(TREND_FILE.read_text()) if TREND_FILE.exists() else {}


def _safe_case_id(value: str) -> str:
    safe = "".join(char for char in value.strip() if char.isalnum() or char in "_-")
    if not safe:
        raise HTTPException(status_code=400, detail="Case ID must contain letters, numbers, _ or -.")
    return safe


# ============================================================
# HELPER: LOAD NIFTI
# ============================================================

def load_nifti(filepath):
    """
    Load a NIfTI file and return its data as a NumPy array.
    """

    # Preserve the dataset-native voxel frame. The trained checkpoint was
    # validated on the native L/P/S arrays, so display and inference must use
    # that same frame.
    nii = nib.load(str(filepath))

    data = nii.get_fdata()

    data = np.asarray(data, dtype=np.float32)

    # Remove NaN / infinite values
    data = np.nan_to_num(data)

    return data


# ============================================================
# HELPER: GENERATE GLB
# ============================================================

def generate_glb_from_volume(volume, output_path):
    """
    Generate a GLB mesh automatically from the uploaded MRI.

    This follows the same basic approach as extract.py:
        NIfTI
          ↓
        Gaussian smoothing
          ↓
        intensity threshold
          ↓
        marching cubes
          ↓
        Trimesh
          ↓
        GLB
    """

    print(f"Generating GLB for {output_path.name}...")

    data = np.asarray(volume, dtype=np.float32)

    data = np.nan_to_num(data)

    # --------------------------------------------------------
    # Smooth MRI
    # --------------------------------------------------------

    smoothed = gaussian_filter(data, sigma=1.0)

    # --------------------------------------------------------
    # Determine threshold
    # --------------------------------------------------------

    max_value = np.max(smoothed)

    if max_value <= 0:
        raise ValueError(
            "MRI volume contains no usable intensity values."
        )

    threshold = max_value * 0.15

    print(
        f"Maximum intensity: {max_value:.3f}"
    )

    print(
        f"Marching cubes threshold: {threshold:.3f}"
    )

    # --------------------------------------------------------
    # Marching cubes
    # --------------------------------------------------------

    vertices, faces, normals, values = measure.marching_cubes(
        smoothed,
        level=threshold
    )

    print(
        f"Generated mesh: "
        f"{len(vertices)} vertices, "
        f"{len(faces)} faces"
    )

    # --------------------------------------------------------
    # Create Trimesh
    # --------------------------------------------------------

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False
    )

    # --------------------------------------------------------
    # Remove unused vertices
    # --------------------------------------------------------

    mesh.remove_unreferenced_vertices()

    # --------------------------------------------------------
    # Remove invalid / zero-area faces
    #
    # This intentionally avoids deprecated Trimesh methods
    # such as remove_degenerate_faces().
    # --------------------------------------------------------

    if len(mesh.faces) > 0:

        try:

            areas = mesh.area_faces

            valid_faces = np.isfinite(areas) & (
                areas > 1e-8
            )

            mesh.update_faces(valid_faces)

        except Exception as e:

            print(
                f"Face cleanup warning: {e}"
            )

    mesh.remove_unreferenced_vertices()

    # --------------------------------------------------------
    # Simplify very large meshes
    # --------------------------------------------------------

    target_faces = 100000

    if len(mesh.faces) > target_faces:

        print(
            f"Simplifying mesh "
            f"from {len(mesh.faces)} faces..."
        )

        try:

            ratio = target_faces / len(mesh.faces)

            mesh = mesh.simplify_quadric_decimation(
                percent=ratio
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

            mesh.apply_translation(
                -mesh.center_mass
            )

        except Exception:

            # Fallback if center_mass cannot be calculated
            center = mesh.vertices.mean(axis=0)

            mesh.apply_translation(-center)

    # --------------------------------------------------------
    # Rotate model to match the frontend orientation
    # --------------------------------------------------------

    rotation = trimesh.transformations.rotation_matrix(
        np.pi,
        [1, 0, 0]
    )

    mesh.apply_transform(rotation)

    # --------------------------------------------------------
    # Export GLB
    # --------------------------------------------------------

    mesh.export(
        str(output_path),
        file_type="glb"
    )

    print(
        f"GLB created successfully: {output_path}"
    )

    return output_path


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "OncoMorph 4D backend"
    }


@app.get("/app", response_class=HTMLResponse)
def frontend():
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return HTMLResponse(FRONTEND_FILE.read_text(encoding="utf-8"))


@app.get("/brain2", response_class=HTMLResponse)
def brain2_frontend():
    """Integrated React UI adapted from Brain2; the source repository is untouched."""
    entry = REACT_UI_DIR / "index.html"
    if not entry.exists():
        raise HTTPException(status_code=404, detail="Integrated Brain2 UI has not been built.")
    return HTMLResponse(entry.read_text(encoding="utf-8"))


@app.post("/api/analyze-glioma-case")
async def analyze_glioma_case(
    case_id: str = Form(...), t1c: UploadFile = File(...), t1n: UploadFile = File(...),
    t2f: UploadFile = File(...), t2w: UploadFile = File(...),
):
    """Run the real four-sequence model and generate matching brain/tumour 3D assets."""
    safe_id = _safe_case_id(case_id)
    case_dir = RAW_DATA_DIR / safe_id
    case_dir.mkdir(parents=True, exist_ok=True)
    uploads = {"t1c": t1c, "t1n": t1n, "t2f": t2f, "t2w": t2w}
    paths = []
    for modality, upload in uploads.items():
        if not upload.filename or not upload.filename.lower().endswith((".nii", ".nii.gz")):
            raise HTTPException(status_code=400, detail=f"{modality} must be a NIfTI file.")
        path = case_dir / f"{modality}.nii.gz"
        with path.open("wb") as destination:
            shutil.copyfileobj(upload.file, destination)
        paths.append(path)
    try:
        mask_path = MASK_DIR / f"{safe_id}_tumor_mask.nii.gz"
        measurements = predict_tumor(paths, mask_path)
        brain_path, tumor_path = DATA_DIR / f"{safe_id}_brain.glb", DATA_DIR / f"{safe_id}_tumor.glb"
        create_brain_glb(str(paths[0]), str(brain_path))
        combined_path = None
        if measurements["tumor_voxels"]:
            if measurements.get("model_type") == "multiclass":
                create_multiclass_tumor_glb(str(mask_path), str(tumor_path))
            else:
                create_tumor_glb(str(mask_path), str(tumor_path))
            combined_path = DATA_DIR / f"{safe_id}_combined.glb"
            create_combined_glb(str(brain_path), str(tumor_path), str(combined_path))
        volume = load_nifti(paths[0])
        mask_volume = (nib.load(str(mask_path)).get_fdata(dtype=np.float32)
                       if measurements["tumor_voxels"] else None)
        scans[safe_id] = {"volume": volume, "mask": mask_volume, "dimensions": list(volume.shape), "model_path": str(brain_path),
                          "source_paths": [str(p) for p in paths] + [str(mask_path)],
                          "tumor_path": str(tumor_path) if measurements["tumor_voxels"] else None,
                          "combined_path": str(combined_path) if combined_path else None,
                          "measurements": measurements}
        trend_history[safe_id] = {"tumor_volume_ml": measurements.get("tumor_volume_ml", 0),
                                  "edema_volume_ml": measurements.get("edema_volume_ml", 0),
                                  "core_volume_ml": measurements.get("core_volume_ml", 0)}
        TREND_FILE.write_text(json.dumps(trend_history, indent=2))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Case analysis failed: {exc}") from exc
    display_coords = np.argwhere(mask_volume > 0) if mask_volume is not None else np.empty((0, 3))
    display_centroid = display_coords.mean(axis=0).round(1).tolist() if len(display_coords) else None
    return {"success": True, "case_id": safe_id, "brain_model_url": f"/api/model/{safe_id}",
            "combined_model_url": f"/api/combined-model/{safe_id}" if measurements["tumor_voxels"] else f"/api/model/{safe_id}",
            "tumor_model_url": f"/api/tumor-model/{safe_id}" if measurements["tumor_voxels"] else None,
            "measurements": {**measurements, "dimensions": list(volume.shape), "display_centroid_voxel": display_centroid}}


@app.get("/api/tumor-model/{scan_id}")
def get_tumor_model(scan_id: str):
    scan = scans.get(scan_id)
    tumor_path = Path(scan["tumor_path"]) if scan and scan.get("tumor_path") else DATA_DIR / f"{scan_id}_tumor.glb"
    if not tumor_path.exists():
        raise HTTPException(status_code=404, detail="Tumour model not found for this case.")
    return FileResponse(str(tumor_path), media_type="model/gltf-binary", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/api/combined-model/{scan_id}")
def get_combined_model(scan_id: str):
    scan = scans.get(scan_id)
    combined_path = Path(scan["combined_path"]) if scan and scan.get("combined_path") else DATA_DIR / f"{scan_id}_combined.glb"
    if not combined_path.exists():
        raise HTTPException(status_code=404, detail="Combined model not found for this case.")
    return FileResponse(str(combined_path), media_type="model/gltf-binary", headers={"Cache-Control": "no-store, no-cache, must-revalidate"})


@app.get("/api/cases/{scan_id}/measurements")
def get_measurements(scan_id: str):
    scan = scans.get(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Case not found.")
    return scan["measurements"]


@app.get("/api/cases/{scan_id}/mask")
def get_case_mask(scan_id: str):
    scan = scans.get(scan_id)
    if not scan or not scan.get("measurements", {}).get("mask_path"):
        raise HTTPException(status_code=404, detail="Prediction mask not found.")
    return FileResponse(scan["measurements"]["mask_path"], media_type="application/gzip",
                        filename=f"{scan_id}_segmentation.nii.gz")


@app.get("/api/cases/{scan_id}/report")
def get_case_report(scan_id: str):
    scan = scans.get(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Case not found.")
    return {"case_id": scan_id, "measurements": scan["measurements"],
            "warning": "Research output only; clinician review required."}


@app.get("/api/patients/{patient_id}/trend")
def get_patient_trend(patient_id: str):
    """Return analyzed longitudinal timepoints for one patient."""
    patient_id = _safe_case_id(patient_id)
    rows = []
    source = {**trend_history, **{scan_id: scan["measurements"] for scan_id, scan in scans.items()}}
    for scan_id, measurements in source.items():
        prefix = patient_id + "_Timepoint_"
        if not scan_id.startswith(prefix):
            continue
        try:
            timepoint = int(scan_id.removeprefix(prefix).split("_")[0])
        except ValueError:
            continue
        m = measurements
        rows.append({"timepoint": timepoint, "case_id": scan_id,
                     "total_ml": m.get("tumor_volume_ml", 0),
                     "edema_ml": m.get("edema_volume_ml", 0),
                     "core_ml": m.get("core_volume_ml", 0),
                     "centroid_voxel": m.get("centroid_voxel") or m.get("display_centroid_voxel"),
                     "class_voxels": m.get("class_voxels", {}),
                     "model_type": m.get("model_type", "multiclass"),
                     "combined_model_url": f"/api/combined-model/{scan_id}" if (DATA_DIR / f"{scan_id}_combined.glb").exists() else None})
    rows.sort(key=lambda row: row["timepoint"])
    for index, row in enumerate(rows):
        previous = rows[index - 1] if index else None
        row["total_change_pct"] = round((row["total_ml"] - previous["total_ml"]) / previous["total_ml"] * 100, 2) if previous and previous["total_ml"] else None
        row["core_change_pct"] = round((row["core_ml"] - previous["core_ml"]) / previous["core_ml"] * 100, 2) if previous and previous["core_ml"] else None
        row["edema_change_pct"] = round((row["edema_ml"] - previous["edema_ml"]) / previous["edema_ml"] * 100, 2) if previous and previous["edema_ml"] else None
    return {"patient_id": patient_id, "timepoints": rows}


@app.post("/api/patients/{patient_id}/analyze-dataset-timeline")
def analyze_dataset_timeline(patient_id: str):
    """Analyze every complete timepoint for a patient from the local dataset."""
    patient_id = _safe_case_id(patient_id)
    patient_dir = DATASET_ROOT / patient_id
    if not patient_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset patient '{patient_id}' not found.")
    analyzed = []
    for timepoint_dir in sorted(patient_dir.glob("Timepoint_*")):
        try:
            timepoint = int(timepoint_dir.name.split("_")[-1])
        except ValueError:
            continue
        paths = [list(timepoint_dir.glob(f"*_brain_{m}.nii.gz")) for m in ("t1c", "t1n", "t2f", "t2w")]
        labels = list(timepoint_dir.glob("*_tumorMask.nii.gz"))
        if not all(paths) or not labels:
            continue
        case_id = f"{patient_id}_Timepoint_{timepoint}"
        mask_path = MASK_DIR / f"{case_id}_tumor_mask.nii.gz"
        measurements = predict_tumor([p[0] for p in paths], mask_path)
        brain_path = DATA_DIR / f"{case_id}_brain.glb"
        tumor_path = DATA_DIR / f"{case_id}_tumor.glb"
        combined_path = DATA_DIR / f"{case_id}_combined.glb"
        create_brain_glb(str(paths[0][0]), str(brain_path))
        if measurements.get("tumor_voxels", 0):
            if measurements.get("model_type") == "multiclass":
                create_multiclass_tumor_glb(str(mask_path), str(tumor_path))
            else:
                create_tumor_glb(str(mask_path), str(tumor_path))
            create_combined_glb(str(brain_path), str(tumor_path), str(combined_path))
        volume = load_nifti(paths[0][0])
        mask_volume = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
        scans[case_id] = {"volume": volume, "mask": mask_volume, "dimensions": list(volume.shape),
                          "source_paths": [str(p[0]) for p in paths] + [str(mask_path)],
                          "model_path": str(brain_path), "tumor_path": str(tumor_path),
                          "combined_path": str(combined_path), "measurements": measurements}
        trend_history[case_id] = {"tumor_volume_ml": measurements.get("tumor_volume_ml", 0),
                                   "edema_volume_ml": measurements.get("edema_volume_ml", 0),
                                   "core_volume_ml": measurements.get("core_volume_ml", 0),
                                   "centroid_voxel": measurements.get("centroid_voxel"),
                                   "class_voxels": measurements.get("class_voxels", {}),
                                   "model_type": measurements.get("model_type", "multiclass")}
        analyzed.append({"case_id": case_id, "timepoint": timepoint, **trend_history[case_id],
                         "combined_model_url": f"/api/combined-model/{case_id}"})
    TREND_FILE.write_text(json.dumps(trend_history, indent=2))
    if not analyzed:
        raise HTTPException(status_code=404, detail="No complete labeled timepoints found.")
    return {"patient_id": patient_id, "analyzed": analyzed, "count": len(analyzed)}


@app.get("/api/patients/{patient_id}/trend.csv")
def export_patient_trend(patient_id: str):
    """Export the longitudinal measurements for review or research records."""
    data = get_patient_trend(patient_id)
    rows = ["patient_id,case_id,timepoint,model,total_abnormal_ml,edema_ml,tumor_core_ml,necrotic_voxels,edema_voxels,non_enhancing_core_voxels,enhancing_tumor_voxels,total_change_pct,edema_change_pct,core_change_pct,centroid_voxel,clinical_note"]
    for item in data["timepoints"]:
        classes = item.get("class_voxels", {})
        values = [data["patient_id"], item["case_id"], item["timepoint"], item.get("model_type", "multiclass"),
                  item["total_ml"], item["edema_ml"], item["core_ml"], classes.get("1", classes.get(1, "")),
                  classes.get("2", classes.get(2, "")), classes.get("3", classes.get(3, "")), classes.get("4", classes.get(4, "")),
                  item.get("total_change_pct"), item.get("edema_change_pct"), item.get("core_change_pct"),
                  item.get("centroid_voxel", ""), "Research output only; clinician review required"]
        rows.append(",".join('"'+str(value).replace('"','""')+'"' if isinstance(value, (list, dict)) or "," in str(value) else ("" if value is None else str(value)) for value in values))
    return Response("\n".join(rows) + "\n", media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{data["patient_id"]}_longitudinal_report.csv"'})


@app.get("/api/patients/{patient_id}/clinical-report.html", response_class=HTMLResponse)
def clinical_report(patient_id: str):
    """Printable research report for clinician review."""
    data = get_patient_trend(patient_id)
    patient = html.escape(data["patient_id"])
    rows = data["timepoints"]
    def change_text(item):
        value = item.get("total_change_pct")
        return "—" if value is None else f"{value:+.2f}%"
    def image_cell(item):
        centroid = item.get("centroid_voxel") or []
        scan = scans.get(item["case_id"])
        if scan and scan.get("mask") is not None:
            mask_coords = np.argwhere(np.asarray(scan["mask"]) > 0)
            if len(mask_coords):
                z = int(round(float(mask_coords[:, 2].mean())))
            else:
                z = int(scan["volume"].shape[2] // 2)
        elif len(centroid) > 2:
            z = int(round(centroid[2]))
        else:
            z = 0
        case_id = html.escape(str(item["case_id"]), quote=True)
        return (f"<img src='/api/slice/axial/{z}?scan_id={case_id}&show_mask=true' "
                f"alt='Timepoint {item['timepoint']} axial MRI with segmentation overlay' "
                "style='width:180px;height:140px;object-fit:contain;background:#000'>")
    body = "".join(
        f"<tr><td>{x['timepoint']}</td><td>{image_cell(x)}</td><td>{x['total_ml']:.2f}</td><td>{x['edema_ml']:.2f}</td>"
        f"<td>{x['core_ml']:.2f}</td><td>{change_text(x)}</td></tr>"
        for x in rows
    )
    latest = rows[-1] if rows else {}
    first = rows[0] if rows else {}
    change = ((latest.get("total_ml", 0) - first.get("total_ml", 0)) / first.get("total_ml", 1) * 100) if rows and first.get("total_ml") else 0
    direction = "increased" if change > 1 else "decreased" if change < -1 else "stable"
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>OncoMorph 4D report - {patient}</title>
    <style>body{{font-family:Arial,sans-serif;color:#172235;max-width:1000px;margin:40px auto;line-height:1.45}}h1{{margin-bottom:4px}}.muted{{color:#5f6f82}}.banner{{padding:14px;background:#fff3cd;border:1px solid #e0bd57;margin:22px 0}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #ccd5df;padding:9px;text-align:left}}th{{background:#e8eef5}}.summary{{display:flex;gap:30px;flex-wrap:wrap}}.box{{padding:12px 18px;background:#f3f6fa;border:1px solid #d7e0ea}}@media print{{button{{display:none}}}}</style></head>
    <body><button onclick='window.print()'>Print / Save as PDF</button><h1>OncoMorph 4D longitudinal report</h1>
    <div class='muted'>Patient: <b>{patient}</b> · Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}</div>
    <div class='banner'><b>Research decision-support output.</b> This report is not a diagnosis, treatment recommendation, or substitute for radiologist/clinician review.</div>
    <h2>Summary</h2><div class='summary'><div class='box'>Timepoints<br><b>{len(rows)}</b></div><div class='box'>Latest total abnormal region<br><b>{latest.get('total_ml', 0):.2f} ml</b></div><div class='box'>Latest edema<br><b>{latest.get('edema_ml', 0):.2f} ml</b></div><div class='box'>Latest tumor core<br><b>{latest.get('core_ml', 0):.2f} ml</b></div><div class='box'>Change from first visit<br><b>{change:+.2f}% ({direction})</b></div></div>
    <h2>Longitudinal measurements and MRI review</h2><p class='muted'>Representative axial image at the predicted lesion centroid. The colored overlay is the model segmentation and must be checked against the full MRI study.</p><table><thead><tr><th>Timepoint</th><th>MRI + overlay</th><th>Total abnormal region (ml)</th><th>Edema (ml)</th><th>Tumor core (ml)</th><th>Change from prior</th></tr></thead><tbody>{body or '<tr><td colspan=6>No analyzed timepoints</td></tr>'}</tbody></table>
    <h2>3D spatial overview</h2><p class='muted'>Red: brain envelope. Green: predicted abnormal region. This is a spatial overview, not a diagnostic rendering.</p>{''.join(f"<h3>Timepoint {x['timepoint']}</h3><img src='/api/cases/{html.escape(str(x['case_id']), quote=True)}/report-3d.png' style='width:520px;max-width:100%;background:#f5f7fa;border:1px solid #ccd5df'><br><a href='/api/cases/{html.escape(str(x['case_id']), quote=True)}/download-all'>Download MRI modalities + tumor mask + measurements (ZIP)</a>" for x in rows)}
    <h2>Interpretation and limitations</h2><ul><li>Edema and tumor-core volumes are reported separately by the multiclass model.</li><li>Increase/decrease flags describe measured segmentation volume only; they do not establish biological progression.</li><li>Review image quality, registration, motion/metal artifacts, and segmentation boundaries on the original MRI.</li><li>Confirm findings with an appropriately qualified clinician before any clinical action.</li></ul>
    </body></html>"""
    return HTMLResponse(document, headers={"Content-Disposition": f'inline; filename="{data["patient_id"]}_clinical_report.html"'})


@app.get("/api/cases/{scan_id}/report-3d.png")
def report_3d_image(scan_id: str):
    """Create a lightweight printable 3D voxel overview for the report."""
    scan = scans.get(scan_id)
    if not scan or scan.get("mask") is None:
        raise HTTPException(status_code=404, detail="3D report data not found; analyze the timeline first.")
    mask = np.asarray(scan["mask"]) > 0
    brain = np.asarray(scan["volume"])
    nonzero = brain[np.isfinite(brain) & (brain > 0)]
    threshold = np.percentile(nonzero, 5) if len(nonzero) else 0
    brain_mask = brain > threshold
    def points(binary, limit):
        coords = np.argwhere(binary)
        if len(coords) > limit:
            coords = coords[np.linspace(0, len(coords) - 1, limit, dtype=int)]
        return coords
    fig = plt.figure(figsize=(7, 6), facecolor="white")
    axis = fig.add_subplot(111, projection="3d")
    def surface(binary, color, alpha, label):
        if not np.any(binary):
            return
        # Downsample only for rendering speed; coordinates remain in voxel space.
        step = max(1, int(round(max(binary.shape) / 150)))
        sampled = binary[::step, ::step, ::step]
        if min(sampled.shape) < 2 or not np.any(sampled):
            return
        vertices, faces, _, _ = measure.marching_cubes(sampled.astype(np.float32), level=0.5)
        vertices *= step
        axis.plot_trisurf(vertices[:, 2], vertices[:, 1], vertices[:, 0], triangles=faces,
                          color=color, alpha=alpha, linewidth=0, antialiased=True, label=label)
    surface(brain_mask, "#d98b68", 0.22, "Brain envelope")
    surface(mask, "#16a34a", 0.82, "Predicted tumor / abnormal region")
    axis.set_xlabel("Axial (z)"); axis.set_ylabel("Y"); axis.set_zlabel("X")
    axis.set_title(f"{scan_id} · registered 3D overview")
    axis.legend(loc="upper right")
    axis.view_init(elev=22, azim=-58)
    fig.tight_layout()
    output = io.BytesIO(); fig.savefig(output, format="png", dpi=150, facecolor="white"); plt.close(fig)
    return Response(output.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/cases/{scan_id}/download-all")
def download_case_files(scan_id: str):
    scan = scans.get(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Case data is not loaded; analyze the timeline first.")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in scan.get("source_paths", []):
            path = Path(source)
            if path.exists():
                archive.write(path, arcname=path.name)
        measurements = json.dumps(scan.get("measurements", {}), indent=2, default=str)
        archive.writestr(f"{scan_id}_measurements.json", measurements)
    output.seek(0)
    return Response(output.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{scan_id}_mri_and_segmentation.zip"'})


# ============================================================
# UPLOAD MRI
# ============================================================

@app.post("/api/upload-mri")
async def upload_mri(
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    scan_id: str = Form(...)
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    # Brain2 sends the four modalities under the repeated `files` field;
    # the legacy viewer sends one `file`. Use T1-native/first volume for the
    # anatomical preview while retaining compatibility with both clients.
    if file is None and files:
        file = files[0]
    if file is None or not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No MRI file provided."
        )

    filename = file.filename

    lower_filename = filename.lower()

    if not (
        lower_filename.endswith(".nii")
        or lower_filename.endswith(".nii.gz")
    ):

        raise HTTPException(
            status_code=400,
            detail="Only NIfTI files (.nii or .nii.gz) are supported."
        )

    # --------------------------------------------------------
    # Clean scan ID
    # --------------------------------------------------------

    scan_id = scan_id.strip()

    if not scan_id:

        raise HTTPException(
            status_code=400,
            detail="Scan ID cannot be empty."
        )

    # Prevent path traversal
    safe_scan_id = "".join(
        c for c in scan_id
        if c.isalnum() or c in "_-"
    )

    if not safe_scan_id:

        raise HTTPException(
            status_code=400,
            detail="Invalid scan ID."
        )

    # Full Brain2 upload path: run the same trained multiclass pipeline used
    # by the OncoMorph workbench instead of creating a brain-only preview.
    if files and len(files) >= 4:
        case_dir = RAW_DATA_DIR / safe_scan_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Brain2 order: T1n, T1c, T2w, FLAIR/T2f.
            modality_names = ("t1n", "t1c", "t2w", "t2f")
            saved = []
            for upload, modality in zip(files[:4], modality_names):
                if not upload.filename or not upload.filename.lower().endswith((".nii", ".nii.gz")):
                    raise HTTPException(status_code=400, detail=f"{modality} must be a NIfTI file.")
                destination = case_dir / f"{modality}.nii.gz"
                with destination.open("wb") as buffer:
                    shutil.copyfileobj(upload.file, buffer)
                saved.append(destination)
            inference_paths = [saved[1], saved[0], saved[3], saved[2]]  # t1c,t1n,t2f,t2w
            mask_path = MASK_DIR / f"{safe_scan_id}_tumor_mask.nii.gz"
            measurements = predict_tumor(inference_paths, mask_path)
            brain_path = DATA_DIR / f"{safe_scan_id}_brain.glb"
            tumor_path = DATA_DIR / f"{safe_scan_id}_tumor.glb"
            combined_path = DATA_DIR / f"{safe_scan_id}_combined.glb"
            create_brain_glb(str(saved[1]), str(brain_path))
            if measurements.get("tumor_voxels", 0):
                if measurements.get("model_type") == "multiclass":
                    create_multiclass_tumor_glb(str(mask_path), str(tumor_path))
                else:
                    create_tumor_glb(str(mask_path), str(tumor_path))
                create_combined_glb(str(brain_path), str(tumor_path), str(combined_path))
            volume = load_nifti(saved[1])
            mask_volume = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
            scans[safe_scan_id] = {"volume": volume, "mask": mask_volume, "dimensions": list(volume.shape),
                                   "model_path": str(brain_path), "tumor_path": str(tumor_path),
                                   "combined_path": str(combined_path), "measurements": measurements,
                                   "source_paths": [str(p) for p in saved] + [str(mask_path)]}
            return {"success": True, "scan_id": safe_scan_id, "filename": files[0].filename,
                    "dimensions": {"x": volume.shape[0], "y": volume.shape[1], "z": volume.shape[2]},
                    "model_url": f"/api/model/{safe_scan_id}", "brain_model_url": f"/api/model/{safe_scan_id}",
                    "tumor_model_url": f"/api/tumor-model/{safe_scan_id}", "mask_url": f"/api/cases/{safe_scan_id}/mask",
                    "measurements": measurements,
                    "tumour_volume_ml": measurements.get("tumor_volume_ml", 0),
                    "tumour_voxels": measurements.get("tumor_voxels", 0),
                    "tumor_volume_ml": measurements.get("tumor_volume_ml", 0),
                    "tumor_voxels": measurements.get("tumor_voxels", 0),
                    "edema_volume_ml": measurements.get("edema_volume_ml", 0),
                    "core_volume_ml": measurements.get("core_volume_ml", 0),
                    "model_type": measurements.get("model_type", "multiclass")}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Multimodal segmentation failed: {exc}") from exc

    # --------------------------------------------------------
    # Save uploaded MRI
    # --------------------------------------------------------

    output_filename = filename

    output_path = RAW_DATA_DIR / output_filename

    # If a file with the same name exists, overwrite it
    with open(output_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    print()
    print("=" * 60)
    print(f"Uploaded MRI: {filename}")
    print(f"Timepoint: {safe_scan_id}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load NIfTI
    # --------------------------------------------------------

    try:

        volume = load_nifti(output_path)

    except Exception as e:

        print(
            f"MRI loading error: {repr(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not read MRI file: {str(e)}"
        )

    # --------------------------------------------------------
    # Validate volume
    # --------------------------------------------------------

    if volume.ndim != 3:

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded MRI must be a 3D NIfTI volume. "
                f"Received {volume.ndim} dimensions."
            )
        )

    dimensions = list(volume.shape)

    print(
        f"MRI dimensions: "
        f"X={dimensions[0]}, "
        f"Y={dimensions[1]}, "
        f"Z={dimensions[2]}"
    )

    # --------------------------------------------------------
    # Generate GLB
    # --------------------------------------------------------

    glb_path = DATA_DIR / f"{safe_scan_id}.glb"

    try:

        generate_glb_from_volume(
            volume,
            glb_path
        )

    except Exception as e:

        print(
            f"MRI processing error: {repr(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"MRI processing failed: {str(e)}"
        )

    # --------------------------------------------------------
    # Store scan in memory
    # --------------------------------------------------------

    scans[safe_scan_id] = {

        "volume": volume,

        "dimensions": dimensions,

        "filename": filename,

        "model_path": str(glb_path),

    }

    print(
        f"Scan {safe_scan_id} ready."
    )

    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Return information to React
    # --------------------------------------------------------

    return {

        "success": True,

        "scan_id": safe_scan_id,

        "filename": filename,

        "dimensions": dimensions,

        "model_url": (
            f"/api/model/{safe_scan_id}"
        ),
        "brain_model_url": f"/api/model/{safe_scan_id}",
        "tumor_model_url": None,
        "mask_url": None,

    }


# ============================================================
# GET DIMENSIONS
# ============================================================

@app.get("/api/dimensions")
def get_dimensions(
    scan_id: str | None = None
):

    # --------------------------------------------------------
    # If a scan ID was provided
    # --------------------------------------------------------

    if scan_id:

        if scan_id not in scans:

            raise HTTPException(
                status_code=404,
                detail="Scan not found."
            )

        dimensions = scans[
            scan_id
        ]["dimensions"]

    # --------------------------------------------------------
    # Otherwise return first available scan
    # --------------------------------------------------------

    elif scans:

        first_scan = next(
            iter(scans.values())
        )

        dimensions = first_scan[
            "dimensions"
        ]

    else:

        raise HTTPException(
            status_code=404,
            detail="No MRI scans loaded."
        )

    return {

        "x": dimensions[0],

        "y": dimensions[1],

        "z": dimensions[2],

    }


# ============================================================
# MATPLOTLIB MRI SLICE
# ============================================================

@app.get("/api/slice/{plane}/{index}")
def get_slice(
    plane: str,
    index: int,
    scan_id: str,
    show_mask: bool = False
):

    # --------------------------------------------------------
    # Verify scan
    # --------------------------------------------------------

    if scan_id not in scans:

        raise HTTPException(
            status_code=404,
            detail=f"Scan '{scan_id}' not found."
        )

    volume = scans[
        scan_id
    ]["volume"]
    mask_volume = scans[scan_id].get("mask")

    # --------------------------------------------------------
    # Get dimensions
    # --------------------------------------------------------

    x_dim = volume.shape[0]

    y_dim = volume.shape[1]

    z_dim = volume.shape[2]

    # --------------------------------------------------------
    # Extract slice
    #
    # Same orientation as the original implementation:
    #
    # Axial   -> Z
    # Coronal -> Y
    # Sagittal -> X
    # --------------------------------------------------------

    if plane == "axial":

        if index < 0 or index >= z_dim:

            raise HTTPException(
                status_code=400,
                detail="Axial slice index out of range."
            )

        slice_2d = volume[:, :, index]
        mask_2d = mask_volume[:, :, index] if mask_volume is not None else None

    elif plane == "coronal":

        if index < 0 or index >= y_dim:

            raise HTTPException(
                status_code=400,
                detail="Coronal slice index out of range."
            )

        slice_2d = volume[:, index, :]
        mask_2d = mask_volume[:, index, :] if mask_volume is not None else None

    elif plane == "sagittal":

        if index < 0 or index >= x_dim:

            raise HTTPException(
                status_code=400,
                detail="Sagittal slice index out of range."
            )

        slice_2d = volume[index, :, :]
        mask_2d = mask_volume[index, :, :] if mask_volume is not None else None

    else:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid plane. "
                "Use axial, coronal, or sagittal."
            )
        )

    # --------------------------------------------------------
    # Rotate exactly as before
    # --------------------------------------------------------

    slice_2d = np.rot90(slice_2d)
    if mask_volume is not None:
        mask_2d = np.rot90(mask_2d)

    # --------------------------------------------------------
    # Matplotlib rendering
    #
    # IMPORTANT:
    # This is deliberately kept as the MRI rendering
    # pipeline rather than replacing it with a browser-side
    # renderer.
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(5, 5),
        dpi=120
    )

    finite = slice_2d[np.isfinite(slice_2d) & (slice_2d > 0)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [1, 99])
        if vmax <= vmin:
            vmin, vmax = float(finite.min()), float(finite.max()) + 1e-6
    else:
        vmin, vmax = 0, 1
    ax.imshow(slice_2d, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    if show_mask and mask_volume is not None and np.any(mask_2d):
        # Multiclass mask colors: edema amber, non-enhancing/necrotic blue,
        # non-enhancing tumor orange, enhancing tumor red.
        class_colors = ListedColormap(["#00000000", "#7b61ffff", "#f2c94cff", "#f2994aff", "#eb5757ff"])
        ax.imshow(np.ma.masked_where(mask_2d <= 0, mask_2d), cmap=class_colors, alpha=0.62, vmin=0, vmax=4)
        ax.contour(mask_2d > 0, levels=[0.5], colors="#ff2020", linewidths=1.2)

    ax.axis("off")

    # Remove margins around the image
    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    # --------------------------------------------------------
    # Save PNG in memory
    # --------------------------------------------------------

    buf = io.BytesIO()

    plt.savefig(
        buf,
        format="PNG",
        bbox_inches="tight",
        pad_inches=0,
        facecolor="black"
    )

    plt.close(fig)

    buf.seek(0)

    # --------------------------------------------------------
    # Return PNG
    # --------------------------------------------------------

    return Response(
        content=buf.getvalue(),
        media_type="image/png"
    )


@app.get("/api/slice/{scan_id}/{plane}/{index}")
def get_slice_brain2(scan_id: str, plane: str, index: int, show_mask: bool = False):
    """Compatibility URL used by the integrated Brain2 React viewer."""
    return get_slice(plane=plane, index=index, scan_id=scan_id, show_mask=show_mask)


# ============================================================
# GET GENERATED GLB
# ============================================================

@app.get("/api/model/{scan_id}")
def get_model(scan_id: str):

    # --------------------------------------------------------
    # Verify scan
    # --------------------------------------------------------

    model_path = Path(scans[scan_id]["model_path"]) if scan_id in scans else DATA_DIR / f"{scan_id}_brain.glb"

    # --------------------------------------------------------
    # Verify file exists
    # --------------------------------------------------------

    if not model_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Generated GLB file not found."
        )

    # --------------------------------------------------------
    # Return GLB
    # --------------------------------------------------------

    return FileResponse(
        path=str(model_path),
        media_type=(
            "model/gltf-binary"
        ),
        filename=model_path.name,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"}
    )


# ============================================================
# OPTIONAL: OLD GLB ROUTE
# ============================================================

@app.get("/api/model/brain/{filename}")
def get_old_model(filename: str):

    """
    Keeps compatibility with the previous frontend/backend
    route if an older component still requests it.
    """

    model_path = DATA_DIR / f"{filename}.glb"

    if not model_path.exists():

        raise HTTPException(
            status_code=404,
            detail="GLB model not found."
        )

    return FileResponse(
        path=str(model_path),
        media_type="model/gltf-binary",
        filename=model_path.name
    )


# ============================================================
# RUN WITH:
#
# python -m uvicorn main:app --reload --port 8000
# ============================================================
