from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_densenet.keras"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.json"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
DISCLAIMER = (
    "Research and education only. This output is not a medical diagnosis and "
    "must be reviewed by a qualified clinician."
)
