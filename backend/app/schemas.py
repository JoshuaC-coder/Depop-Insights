from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ListingInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(default="")
    description: str = Field(default="")
    price: float | None = Field(default=None, ge=0)
    shipping_price: float | None = Field(default=None, ge=0)
    total_buyer_cost: float | None = Field(default=None, ge=0)
    brand: str | None = None
    category: str | None = None
    hoodie_type: str | None = None
    size: str | None = None
    condition: str | None = None
    color: str | None = None
    likes: int | None = Field(default=None, ge=0)
    comments_count: int | None = Field(default=None, ge=0)
    seller_rating: float | None = Field(default=None, ge=0, le=5)
    seller_followers: int | None = Field(default=None, ge=0)
    discounted: bool | None = None
    listed_date: datetime | None = None
    sold_date: datetime | None = None
    image_urls: list[HttpUrl] | list[str] = Field(default_factory=list)
    sold_status: bool | None = None
    source_url: str | None = None


class Factor(BaseModel):
    label: str
    impact: str
    detail: str


class PredictResponse(BaseModel):
    fast_sell_score: int
    sold_within_30_days_probability: float
    estimated_days_to_sell: dict[str, int]
    suggested_price_range: dict[str, float]
    top_strengths: list[str]
    top_weaknesses: list[str]
    actionable_recommendations: list[str]
    top_factors: list[Factor]
    model_status: str
    model_source: str
    notes: list[str] = Field(default_factory=list)
    raw_predictions: dict[str, Any] = Field(default_factory=dict)
