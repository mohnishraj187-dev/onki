from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"

UPLOAD_DIR = DATA_DIR / "uploads"
MASK_DIR = DATA_DIR / "masks"
MODEL_OUTPUT_DIR = DATA_DIR / "models"

# Put the checkpoint produced by your training code here.
MODEL_CHECKPOINT = BACKEND_DIR / "models" / "tumor_unet_epoch_5.pth"

for directory in (UPLOAD_DIR, MASK_DIR, MODEL_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = (".nii", ".nii.gz")
