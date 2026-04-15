from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_FILENAMES = {
    "classifier": "classifier.joblib",
    "days_regressor": "days_regressor.joblib",
    "price_regressor": "price_regressor.joblib",
    "metadata": "metadata.joblib",
}

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
]
