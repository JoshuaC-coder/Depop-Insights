from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backend.app.schemas import ListingInput, PredictResponse
from backend.app.services.model_registry import ModelBundle
from backend.app.services.recommendations import build_factors
from ml.depop_insights_ml.inference import build_inference_frame


@dataclass
class PredictionResult:
    probability: float
    expected_days: float
    expected_price: float
    model_status: str
    model_source: str
    notes: list[str]
    raw_predictions: dict[str, Any]


def _bounded_probability(value: float) -> float:
    return max(0.02, min(0.98, value))


def _estimate_price_band(center: float) -> dict[str, float]:
    low = max(0.0, round(center * 0.92, 2))
    high = round(center * 1.08, 2)
    return {"low": low, "high": high}


def _estimate_day_band(center: float) -> dict[str, int]:
    return {
        "low": max(1, int(round(center * 0.8))),
        "high": max(2, int(round(center * 1.2))),
    }


def _heuristic_predict(listing: ListingInput) -> PredictionResult:
    title = listing.title.lower()
    description = listing.description.lower()
    total_cost = listing.total_buyer_cost or (listing.price or 0) + (listing.shipping_price or 0)
    score = 0.48

    positive_terms = ["hoodie", "crewneck", "zip", "oversized", "vintage", "graphic", "heavyweight"]
    score += min(0.16, sum(term in title for term in positive_terms) * 0.03)
    score += min(0.08, sum(term in description for term in positive_terms) * 0.02)

    if listing.brand:
        score += 0.05
    if listing.likes is not None:
        score += min(0.12, listing.likes * 0.01)
    if listing.shipping_price is not None and listing.shipping_price > 10:
        score -= 0.07
    if total_cost > 75:
        score -= 0.08
    if len(listing.description.split()) < 12:
        score -= 0.05
    if listing.condition and listing.condition.lower() in {"new", "like new", "excellent"}:
        score += 0.04

    probability = _bounded_probability(score)
    expected_days = max(5.0, round(55 - (probability * 40), 1))
    expected_price = round(listing.price or max(25.0, total_cost * 0.92), 2)

    return PredictionResult(
        probability=probability,
        expected_days=expected_days,
        expected_price=expected_price,
        model_status="heuristic_fallback",
        model_source="rules_based_baseline",
        notes=[
            "No trained resale pricing artifacts were found in data/models.",
            "Both price guidance and sell-speed outputs are currently heuristic placeholders.",
        ],
        raw_predictions={"heuristic_probability": probability, "heuristic_days": expected_days, "heuristic_price": expected_price},
    )


def _model_predict(bundle: ModelBundle, listing: ListingInput) -> PredictionResult:
    frame = build_inference_frame(pd.DataFrame([listing.model_dump()]))
    classifier = bundle.classifier

    if classifier is not None:
        probability = float(classifier.predict_proba(frame)[0][1]) if hasattr(classifier, "predict_proba") else float(classifier.predict(frame)[0])
        model_status = "trained_model"
        model_source = str(bundle.metadata.get("model_family", "pipeline"))
        notes: list[str] = []
    else:
        heuristic = _heuristic_predict(listing)
        probability = heuristic.probability
        model_status = "hybrid_price_model"
        model_source = str(bundle.metadata.get("model_family", "price_only_pipeline"))
        notes = [
            "The current trained artifacts come from a price-labeled dataset without sell-through outcome labels.",
            "Fast-sell probability and estimated days are still heuristic, while price guidance is model-driven.",
        ]

    days_model = bundle.days_regressor
    price_model = bundle.price_regressor
    expected_days = float(days_model.predict(frame)[0]) if days_model is not None else max(5.0, round(55 - (probability * 40), 1))
    expected_price = float(price_model.predict(frame)[0]) if price_model is not None else float(listing.price or 35.0)

    trained_at = bundle.metadata.get("trained_at")
    if trained_at:
        notes.append(f"Using trained artifacts generated at {trained_at}.")

    return PredictionResult(
        probability=_bounded_probability(probability),
        expected_days=max(1.0, expected_days),
        expected_price=max(0.0, expected_price),
        model_status=model_status,
        model_source=model_source,
        notes=notes,
        raw_predictions={
            "probability": probability,
            "expected_days": expected_days,
            "expected_price": expected_price,
        },
    )


def predict_listing(listing: ListingInput, bundle: ModelBundle) -> PredictResponse:
    prediction = _model_predict(bundle, listing) if bundle.is_ready else _heuristic_predict(listing)
    factors, strengths, weaknesses, suggestions = build_factors(
        listing=listing,
        probability=prediction.probability,
        expected_days=prediction.expected_days,
    )

    fast_sell_score = int(round(prediction.probability * 100))
    return PredictResponse(
        fast_sell_score=fast_sell_score,
        sold_within_30_days_probability=round(prediction.probability, 4),
        estimated_days_to_sell=_estimate_day_band(prediction.expected_days),
        suggested_price_range=_estimate_price_band(prediction.expected_price),
        top_strengths=strengths,
        top_weaknesses=weaknesses,
        actionable_recommendations=suggestions,
        top_factors=factors,
        model_status=prediction.model_status,
        model_source=prediction.model_source,
        notes=prediction.notes,
        raw_predictions=prediction.raw_predictions,
    )
