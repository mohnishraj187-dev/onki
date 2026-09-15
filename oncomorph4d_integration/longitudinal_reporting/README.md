# OncoMorph 4D longitudinal reporting

This additive module documents and packages the longitudinal features used by
the port-8000 workbench:

- patient/timepoint timeline analysis
- total abnormal-region, edema, and tumor-core volumes
- percentage change from the previous timepoint
- longitudinal volume chart
- printable clinical-style HTML report
- CSV export
- representative MRI slice with segmentation overlay
- registered 3D report image
- ZIP download of all four MRI modalities, predicted mask, and measurements

The production routes are currently implemented in the OncoMorph FastAPI
application. The endpoint contract is:

```text
POST /api/patients/{patient_id}/analyze-dataset-timeline
GET  /api/patients/{patient_id}/trend
GET  /api/patients/{patient_id}/trend.csv
GET  /api/patients/{patient_id}/clinical-report.html
GET  /api/cases/{scan_id}/report-3d.png
GET  /api/cases/{scan_id}/download-all
```

All outputs are research decision-support results and require clinician
review. Volume changes do not independently establish biological progression.
