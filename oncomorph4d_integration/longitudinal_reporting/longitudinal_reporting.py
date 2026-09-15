"""Small standalone helpers for exporting OncoMorph 4D trend data.

The FastAPI application uses equivalent logic in its reporting routes. This
file lets another service create a CSV or a compact HTML report from the JSON
returned by GET /api/patients/{patient_id}/trend.
"""
from __future__ import annotations

import csv
import html
import io
from datetime import datetime


def trend_csv(trend: dict) -> str:
    """Return a detailed longitudinal CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "patient_id", "case_id", "timepoint", "model", "total_abnormal_ml",
        "edema_ml", "tumor_core_ml", "total_change_pct", "edema_change_pct",
        "core_change_pct", "centroid_voxel", "clinical_note",
    ])
    for item in trend.get("timepoints", []):
        writer.writerow([
            trend.get("patient_id", ""), item.get("case_id", ""),
            item.get("timepoint", ""), item.get("model_type", "multiclass"),
            item.get("total_ml", 0), item.get("edema_ml", 0),
            item.get("core_ml", 0), item.get("total_change_pct", ""),
            item.get("edema_change_pct", ""), item.get("core_change_pct", ""),
            item.get("centroid_voxel", ""),
            "Research output only; clinician review required",
        ])
    return output.getvalue()


def clinical_html(trend: dict) -> str:
    """Return a printable, non-diagnostic HTML summary."""
    patient = html.escape(str(trend.get("patient_id", "unknown")))
    rows = trend.get("timepoints", [])
    table = []
    for item in rows:
        change = item.get("total_change_pct")
        change_text = "—" if change is None else f"{change:+.2f}%"
        table.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('timepoint', '')))}</td>"
            f"<td>{float(item.get('total_ml', 0)):.2f}</td>"
            f"<td>{float(item.get('edema_ml', 0)):.2f}</td>"
            f"<td>{float(item.get('core_ml', 0)):.2f}</td>"
            f"<td>{change_text}</td></tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>OncoMorph 4D report - {patient}</title>
<style>body{{font-family:Arial;max-width:950px;margin:36px auto;color:#172235}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5df;padding:8px}}th{{background:#e8eef5}}.warning{{background:#fff3cd;padding:14px}}</style></head>
<body><button onclick="window.print()">Print / Save as PDF</button>
<h1>OncoMorph 4D longitudinal report</h1>
<p>Patient: <b>{patient}</b><br>Generated: {datetime.now().isoformat(timespec='seconds')}</p>
<p class="warning"><b>Research decision-support output.</b> Clinician review is required.</p>
<h2>Measurements</h2><table><thead><tr><th>Timepoint</th><th>Total abnormal (ml)</th><th>Edema (ml)</th><th>Tumor core (ml)</th><th>Change</th></tr></thead>
<tbody>{''.join(table) or '<tr><td colspan="5">No analyzed timepoints</td></tr>'}</tbody></table>
<h2>Limitations</h2><p>Volume change alone does not establish biological progression. Review the original MRI, registration, artifacts, and segmentation boundaries.</p>
</body></html>"""
