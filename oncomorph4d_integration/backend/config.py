from pathlib import Path


# ---------------------------------------------------------
# BASE DIRECTORIES
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"

GENERATED_DIR = BASE_DIR / "generated"

MASK_DIR = GENERATED_DIR / "masks"

MODEL_DIR = GENERATED_DIR / "models"

CHECKPOINT_DIR = BASE_DIR / "models"


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

MODEL_CHECKPOINT = CHECKPOINT_DIR / "glioma_segmentation_best_model.pt"
MULTICLASS_MODEL_CHECKPOINT = BASE_DIR.parent / "model" / "best_validation_model.pt"


# ---------------------------------------------------------
# CREATE DIRECTORIES
# ---------------------------------------------------------

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
