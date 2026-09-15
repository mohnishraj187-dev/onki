# OncoMorph 4D multiclass model integration

This folder is an additive integration bundle for Brain2. Existing Brain2
files are intentionally unchanged.

Contents:

- `model/best_model.pt`: trained multiclass SegResNet checkpoint
- `backend/segmentation_service.py`: inference and volume measurements
- `backend/mesh_service.py`: registered brain/tumor GLB generation
- `backend/config.py`: model and storage configuration

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
