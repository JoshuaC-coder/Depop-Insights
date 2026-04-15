from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from backend.app.config import MODEL_FILENAMES, MODELS_DIR


@dataclass
class ModelBundle:
    classifier: Any | None = None
    days_regressor: Any | None = None
    price_regressor: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return any(model is not None for model in (self.classifier, self.days_regressor, self.price_regressor))


def _load_artifact(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def load_models(models_dir: Path = MODELS_DIR) -> ModelBundle:
    return ModelBundle(
        classifier=_load_artifact(models_dir / MODEL_FILENAMES["classifier"]),
        days_regressor=_load_artifact(models_dir / MODEL_FILENAMES["days_regressor"]),
        price_regressor=_load_artifact(models_dir / MODEL_FILENAMES["price_regressor"]),
        metadata=_load_artifact(models_dir / MODEL_FILENAMES["metadata"]) or {},
    )
