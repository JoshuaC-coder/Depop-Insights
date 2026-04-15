from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import DEFAULT_ALLOWED_ORIGINS
from backend.app.schemas import ListingInput, PredictResponse
from backend.app.services.model_registry import ModelBundle, load_models
from backend.app.services.predictor import predict_listing


app = FastAPI(title="Depop Insights API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ALLOWED_ORIGINS,
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_bundle: ModelBundle = load_models()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_status": "trained_model" if model_bundle.is_ready else "heuristic_fallback",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(listing: ListingInput) -> PredictResponse:
    return predict_listing(listing, model_bundle)
