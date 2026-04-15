from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.depop_insights_ml.data_schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


TREND_KEYWORDS = [
    "vintage",
    "y2k",
    "oversized",
    "faded",
    "distressed",
    "rare",
    "streetwear",
    "heavyweight",
    "essential",
    "graphic",
]


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["title"] = enriched["title"].fillna("")
    enriched["description"] = enriched["description"].fillna("")
    enriched["title_length"] = enriched["title"].str.len()
    enriched["description_length"] = enriched["description"].str.len()
    enriched["title_word_count"] = enriched["title"].str.split().str.len()
    enriched["description_word_count"] = enriched["description"].str.split().str.len()
    enriched["brand_mentioned_in_title"] = [
        int(bool(brand) and str(brand).lower() in title.lower())
        for title, brand in zip(enriched["title"], enriched["brand"].fillna(""))
    ]
    enriched["keyword_count_title"] = sum(enriched["title"].str.contains(keyword, case=False, regex=False) for keyword in TREND_KEYWORDS)
    enriched["keyword_count_description"] = sum(enriched["description"].str.contains(keyword, case=False, regex=False) for keyword in TREND_KEYWORDS)
    enriched["condition_transparency"] = enriched["description"].str.contains(r"flaw|stain|faded|excellent|good|worn", case=False, regex=True).astype(int)
    enriched["listed_day_of_week"] = enriched["listed_date"].dt.dayofweek.fillna(-1)
    enriched["listed_month"] = enriched["listed_date"].dt.month.fillna(0)
    enriched["listed_season"] = enriched["listed_month"].map({12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}).fillna("unknown")

    group_brand = enriched.groupby(["brand", "category"], dropna=False)
    enriched["brand_category_median_price"] = group_brand["price"].transform("median")
    enriched["brand_category_avg_likes"] = group_brand["likes"].transform("mean")
    enriched["competition_count"] = group_brand["title"].transform("count")
    enriched["price_percentile_like_group"] = group_brand["price"].rank(pct=True)
    enriched["underpriced_indicator"] = (enriched["price"] < enriched["brand_category_median_price"]).fillna(False).astype(int)
    enriched["overpriced_indicator"] = (enriched["price"] > enriched["brand_category_median_price"]).fillna(False).astype(int)
    return enriched


def _merge_text_columns(frame: pd.DataFrame) -> np.ndarray:
    title = frame["title"].fillna("")
    description = frame["description"].fillna("")
    return (title + " " + description).to_numpy()


def build_feature_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    enriched = add_engineered_features(frame)
    numeric_features = NUMERIC_COLUMNS + [
        "title_length",
        "description_length",
        "title_word_count",
        "description_word_count",
        "brand_mentioned_in_title",
        "keyword_count_title",
        "keyword_count_description",
        "condition_transparency",
        "listed_day_of_week",
        "listed_month",
        "brand_category_median_price",
        "brand_category_avg_likes",
        "competition_count",
        "price_percentile_like_group",
        "underpriced_indicator",
        "overpriced_indicator",
    ]
    categorical_features = CATEGORICAL_COLUMNS + ["listed_season"]
    numeric_features = [column for column in numeric_features if column in enriched.columns and enriched[column].notna().any()]
    categorical_features = [column for column in categorical_features if column in enriched.columns and enriched[column].notna().any()]
    return enriched, numeric_features, categorical_features


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    text_pipeline = Pipeline(
        steps=[
            ("selector", FunctionTransformer(_merge_text_columns, validate=False)),
            ("tfidf", TfidfVectorizer(max_features=300, ngram_range=(1, 2), min_df=1)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("text", text_pipeline, ["title", "description"]),
        ]
    )
