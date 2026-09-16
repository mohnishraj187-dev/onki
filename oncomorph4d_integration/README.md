# OncoMorph 4D multiclass model integration

This folder is an additive integration bundle for Brain2. Existing Brain2
files are intentionally unchanged.

Contents:

- `model/best_validation_model.pt`: best-validation multiclass SegResNet checkpoint used by inference
- `backend/segmentation_service.py`: inference and volume measurements
- `backend/mesh_service.py`: registered brain/tumor GLB generation
- `backend/config.py`: model and storage configuration
- `inference.py`: standalone command-line inference entry point
- `requirements.txt`: runtime dependencies

The model predicts background plus four segmentation classes:

1. necrotic / non-enhancing
2. edema
3. non-enhancing core
4. enhancing tumor

The checkpoint is research output only and requires clinician review before
any clinical interpretation or action.

The current Brain2 API can adopt this bundle by importing the service from
this directory and using the existing upload flow to pass the four aligned
NIfTI modalities in the order T1c, T1n, T2-FLAIR, and T2-weighted.

## Standalone inference

From this directory, install the requirements and run:

```bash
pip install -r requirements.txt
python inference.py \
  --t1c /path/to/t1c.nii.gz \
  --t1n /path/to/t1n.nii.gz \
  --flair /path/to/flair.nii.gz \
  --t2w /path/to/t2w.nii.gz \
  --output-mask outputs/tumor_mask.nii.gz \
  --output-json outputs/measurements.json
```

The checkpoint is loaded from `model/best_validation_model.pt`; no machine-specific
absolute path is required.
