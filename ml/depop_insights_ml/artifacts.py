from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from backend.app.config import MODEL_FILENAMES


def save_artifact(artifact: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def save_model_bundle(models_dir: Path, classifier: Any, days_regressor: Any | None, price_regressor: Any | None, metadata: dict[str, Any]) -> None:
    save_artifact(classifier, models_dir / MODEL_FILENAMES["classifier"])
    if days_regressor is not None:
        save_artifact(days_regressor, models_dir / MODEL_FILENAMES["days_regressor"])
    if price_regressor is not None:
        save_artifact(price_regressor, models_dir / MODEL_FILENAMES["price_regressor"])

    metadata = {
        **metadata,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    save_artifact(metadata, models_dir / MODEL_FILENAMES["metadata"])
