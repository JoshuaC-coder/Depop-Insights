from __future__ import annotations

import pandas as pd

from ml.depop_insights_ml.feature_engineering import add_engineered_features
from ml.depop_insights_ml.preprocess import coerce_types, normalize_columns


def build_inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_columns(frame)
    coerced = coerce_types(normalized)
    enriched = add_engineered_features(coerced)
    return enriched
