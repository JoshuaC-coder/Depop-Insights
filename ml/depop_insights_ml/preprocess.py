from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.depop_insights_ml.data_schema import CANONICAL_COLUMNS, COLUMN_ALIASES, TargetColumns


def _to_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    truthy = {"true", "1", "yes", "y"}
    falsy = {"false", "0", "no", "n"}

    def convert(value: object) -> bool:
        if pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        return bool(value)

    return series.apply(convert)


def load_raw_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    read_kwargs: dict[str, object] = {}
    if path.suffix.lower() == ".tsv":
        read_kwargs["sep"] = "\t"
    return pd.read_csv(path, **read_kwargs)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns={source: target for source, target in COLUMN_ALIASES.items() if source in frame.columns}).copy()
    for column in CANONICAL_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = pd.NA
    return renamed


def coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    numeric_columns = [
        "price",
        "shipping_price",
        "total_buyer_cost",
        "likes",
        "comments_count",
        "seller_rating",
        "seller_followers",
    ]
    date_columns = ["listed_date", "sold_date"]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "shipping_price" in cleaned.columns:
        shipping = cleaned["shipping_price"]
        cleaned["shipping_price"] = shipping.where(~shipping.isin([0, 1]), shipping.astype(float) * 7.99)

    for column in date_columns:
        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    cleaned["discounted"] = _to_bool_series(cleaned["discounted"], default=False)
    sold_default = cleaned["sold_date"].notna()
    sold_status = cleaned["sold_status"].copy()
    sold_status_values = [fallback if pd.isna(value) else value for value, fallback in zip(sold_status.tolist(), sold_default.tolist())]
    cleaned["sold_status"] = _to_bool_series(pd.Series(sold_status_values, index=cleaned.index), default=False)
    cleaned["total_buyer_cost"] = cleaned["total_buyer_cost"].fillna(cleaned["price"].fillna(0) + cleaned["shipping_price"].fillna(0))
    return cleaned


def derive_targets(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.copy()
    targets = TargetColumns()
    labeled[targets.days_to_sell] = (labeled["sold_date"] - labeled["listed_date"]).dt.days
    sold_within_30_days = labeled[targets.days_to_sell].le(30)
    labeled[targets.sold_within_30_days] = sold_within_30_days.where(labeled[targets.days_to_sell].notna(), pd.NA)
    labeled[targets.expected_sale_price] = labeled["price"]
    return labeled


def filter_hoodie_mvp(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    hoodie_pattern = r"hoodie|crewneck|crew neck|zip[- ]?up|sweatshirt"
    title_mask = filtered["title"].fillna("").str.contains(hoodie_pattern, case=False, regex=True)
    category_mask = filtered["category"].fillna("").str.contains(hoodie_pattern, case=False, regex=True)
    type_mask = filtered["hoodie_type"].fillna("").str.contains(hoodie_pattern, case=False, regex=True)
    return filtered[title_mask | category_mask | type_mask].reset_index(drop=True)


def prepare_dataset(csv_path: str | Path, hoodie_only: bool = True) -> pd.DataFrame:
    frame = load_raw_csv(csv_path)
    frame = normalize_columns(frame)
    frame = coerce_types(frame)
    frame = derive_targets(frame)
    if hoodie_only:
        frame = filter_hoodie_mvp(frame)
    return frame
