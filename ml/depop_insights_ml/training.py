from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from backend.app.config import MODELS_DIR
from ml.depop_insights_ml.artifacts import save_model_bundle
from ml.depop_insights_ml.data_schema import TargetColumns
from ml.depop_insights_ml.feature_engineering import build_feature_columns, build_preprocessor
from ml.depop_insights_ml.preprocess import prepare_dataset

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
    CatBoostRegressor = None


@dataclass
class TrainingArtifacts:
    classifier: Any
    days_regressor: Any | None
    price_regressor: Any | None
    metrics: dict[str, Any]
    dataset_rows: int


def _label_readiness_summary(dataset: pd.DataFrame, targets: TargetColumns) -> dict[str, Any]:
    sold_within_30_days = pd.Series(dataset[targets.sold_within_30_days])
    return {
        "dataset_rows": len(dataset),
        "classification_classes": sorted(sold_within_30_days.dropna().unique().tolist()),
        "classified_positive_rows": int(sold_within_30_days.eq(1).sum()),
        "classified_negative_rows": int(sold_within_30_days.eq(0).sum()),
        "days_to_sell_non_null": int(dataset[targets.days_to_sell].notna().sum()),
        "listed_date_non_null": int(dataset["listed_date"].notna().sum()),
        "sold_date_non_null": int(dataset["sold_date"].notna().sum()),
        "sold_status_true_rows": int(dataset["sold_status"].fillna(False).astype(bool).sum()),
    }


def _build_classifier(preprocessor: Any, use_catboost: bool = True) -> Pipeline:
    if use_catboost and CatBoostClassifier is not None:
        model = CatBoostClassifier(verbose=0, depth=6, learning_rate=0.05, iterations=250, loss_function="Logloss")
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _build_regressor(preprocessor: Any, use_catboost: bool = True) -> Pipeline:
    if use_catboost and CatBoostRegressor is not None:
        model = CatBoostRegressor(verbose=0, depth=6, learning_rate=0.05, iterations=300, loss_function="RMSE")
    else:
        model = RandomForestRegressor(n_estimators=250, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _classification_metrics(y_true: pd.Series, probabilities: list[float], predictions: list[int]) -> dict[str, Any]:
    return {
        "accuracy": round(accuracy_score(y_true, predictions), 4),
        "precision": round(precision_score(y_true, predictions, zero_division=0), 4),
        "recall": round(recall_score(y_true, predictions, zero_division=0), 4),
        "f1": round(f1_score(y_true, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probabilities), 4) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def _regression_metrics(y_true: pd.Series, predictions: list[float]) -> dict[str, Any]:
    mse = mean_squared_error(y_true, predictions)
    return {
        "mae": round(mean_absolute_error(y_true, predictions), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(r2_score(y_true, predictions), 4),
    }


def train_all(csv_path: str | Path, models_dir: Path = MODELS_DIR, hoodie_only: bool = False, use_catboost: bool = True) -> TrainingArtifacts:
    dataset = prepare_dataset(csv_path, hoodie_only=hoodie_only)
    if dataset.empty:
        raise ValueError("No rows matched the active category filter. Add more data or disable niche filtering before training.")

    targets = TargetColumns()
    readiness = _label_readiness_summary(dataset, targets)
    dataset, numeric_features, categorical_features = build_feature_columns(dataset)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    feature_columns = ["title", "description", *numeric_features, *categorical_features]
    X = dataset[feature_columns]
    y_class = dataset[targets.sold_within_30_days]
    y_days = dataset[targets.days_to_sell]
    y_price = dataset[targets.expected_sale_price]
    metrics: dict[str, Any] = {"label_readiness": readiness}

    price_dataset = X.loc[y_price.notna()]
    y_price_non_null = y_price.loc[y_price.notna()]
    if len(price_dataset) < 8:
        raise ValueError(
            "Cannot train price model because there are fewer than 8 labeled price rows. "
            f"Label summary: {readiness}."
        )

    X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(
        price_dataset,
        y_price_non_null,
        test_size=0.25,
        random_state=42,
    )

    classifier = None
    classification_dataset = X.loc[y_class.notna()]
    y_class_non_null = y_class.loc[y_class.notna()]
    if y_class_non_null.nunique() >= 2:
        stratify = y_class_non_null if y_class_non_null.nunique() > 1 else None
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            classification_dataset,
            y_class_non_null,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
        classifier = _build_classifier(preprocessor, use_catboost=use_catboost)
        classifier.fit(X_train_class, y_train_class)

        class_predictions = classifier.predict(X_test_class)
        class_probabilities = classifier.predict_proba(X_test_class)[:, 1] if hasattr(classifier, "predict_proba") else class_predictions
        metrics["classification"] = _classification_metrics(y_test_class, class_probabilities, class_predictions)
    else:
        metrics["classification"] = {
            "status": "skipped",
            "reason": "Dataset does not contain at least two non-null classes for sold_within_30_days.",
        }

    days_regressor = None
    days_dataset = X.loc[y_days.notna()]
    y_days_non_null = y_days.loc[y_days.notna()]
    if len(days_dataset) >= 8:
        X_train_days, X_test_days, y_train_days, y_test_days = train_test_split(
            days_dataset,
            y_days_non_null,
            test_size=0.25,
            random_state=42,
        )
        days_regressor = _build_regressor(preprocessor, use_catboost=use_catboost)
        days_regressor.fit(X_train_days, y_train_days)
        day_predictions = days_regressor.predict(X_test_days)
        metrics["days_regression"] = _regression_metrics(y_test_days, day_predictions)
    else:
        metrics["days_regression"] = {
            "status": "skipped",
            "reason": "Dataset does not contain enough non-null days_to_sell labels.",
        }

    price_regressor = _build_regressor(preprocessor, use_catboost=use_catboost)
    price_regressor.fit(X_train_price, y_train_price)
    price_predictions = price_regressor.predict(X_test_price)
    metrics["price_regression"] = _regression_metrics(y_test_price, price_predictions)

    save_model_bundle(
        models_dir=models_dir,
        classifier=classifier,
        days_regressor=days_regressor,
        price_regressor=price_regressor,
        metadata={
            "model_family": "catboost_pipeline" if use_catboost and CatBoostClassifier is not None else "sklearn_baseline_pipeline",
            "hoodie_only": hoodie_only,
            "dataset_rows": len(dataset),
            "feature_columns": feature_columns,
            "metrics": metrics,
            "training_scope": "mercari_price_first" if classifier is None else "full_multitask",
        },
    )

    return TrainingArtifacts(
        classifier=classifier,
        days_regressor=days_regressor,
        price_regressor=price_regressor,
        metrics=metrics,
        dataset_rows=len(dataset),
    )
