# OncoMorph 4D Backend

## Pipeline

Uploaded MRI
    |
    +--> brain GLB
    |
    +--> trained SimpleUNet, slice by slice
             |
             +--> 3D tumour mask NIfTI
                       |
                       +--> tumour GLB
    |
    +--> Matplotlib axial/coronal/sagittal PNGs

## Your exact trained model

The supplied training model is a 2D SimpleUNet with:
- 4 input channels
- 1 output channel
- DoubleConv blocks
- MaxPool2d
- ConvTranspose2d
- BatchNorm2d
- ReLU

The supplied train.py saves a state_dict at:
weights/tumor_unet_epoch_5.pth

Copy that file to:

backend/models/tumor_unet_epoch_5.pth

## Important single-MRI limitation

Your training dataset trains on four modalities:
T1n, T1c, T2w, T2f.

Your requested frontend uploads one MRI file.

Therefore the backend runs the exact SimpleUNet slice-by-slice but repeats
the uploaded modality into all four model channels.

This is an engineering bridge for the hackathon workflow. It is not the same
as inference with four true modalities and should not be presented as clinically
validated performance.

If you later want true four-modality inference, the frontend/backend should
accept T1n, T1c, T2w and T2f for each timepoint.

## Run

python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

Health:
http://localhost:8000/api/health
