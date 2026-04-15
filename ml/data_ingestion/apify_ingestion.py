from __future__ import annotations

import json
import os
import re
from urllib.parse import quote_plus
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from ml.depop_insights_ml.data_schema import CANONICAL_COLUMNS
from ml.depop_insights_ml.preprocess import normalize_columns


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "raw"
HOODIE_PATTERN = re.compile(r"hoodie|crewneck|crew neck|zip[- ]?up|sweatshirt", re.IGNORECASE)


@dataclass(frozen=True)
class ApifyConfig:
    api_token: str
    actor_id: str
    query: str
    max_items: int
    api_base_url: str = "https://api.apify.com/v2"
    actor_input_path: str | None = None
    poll_seconds: int = 5
    max_attempts: int = 360


class ApifyRequestError(RuntimeError):
    """Raised when Apify rejects or fails a request with useful context."""


def load_apify_config(env_path: str | Path | None = None) -> ApifyConfig:
    load_dotenv(dotenv_path=env_path)
    api_token = os.getenv("APIFY_API_TOKEN", "").strip()
    actor_id = os.getenv("APIFY_ACTOR_ID", "").strip()
    query = os.getenv("APIFY_SEARCH_QUERY", "hoodie sweatshirt crewneck zip up oversized sweatshirt").strip()
    max_items = int(os.getenv("APIFY_MAX_ITEMS", "200"))
    actor_input_path = os.getenv("APIFY_ACTOR_INPUT_PATH", "").strip() or None
    poll_seconds = int(os.getenv("APIFY_POLL_SECONDS", "5"))
    max_attempts = int(os.getenv("APIFY_MAX_ATTEMPTS", "360"))

    if not api_token:
        raise ValueError("APIFY_API_TOKEN is missing. Add it to your .env file before fetching data.")
    if not actor_id:
        raise ValueError("APIFY_ACTOR_ID is missing. Add it to your .env file before fetching data.")

    return ApifyConfig(
        api_token=api_token,
        actor_id=actor_id,
        query=query,
        max_items=max_items,
        actor_input_path=actor_input_path,
        poll_seconds=poll_seconds,
        max_attempts=max_attempts,
    )


def _build_actor_input(config: ApifyConfig) -> dict[str, Any]:
    search_terms = [
        "hoodie",
        "crewneck",
        "zip up hoodie",
        "oversized sweatshirt",
    ]
    depop_search_url = f"https://www.depop.com/search/?q={quote_plus(config.query)}"
    base_input: dict[str, Any] = {
        "query": config.query,
        "queries": [config.query],
        "search": config.query,
        "searchTerms": search_terms,
        "searchStringsArray": search_terms,
        "keyword": config.query,
        "keywords": search_terms,
        "startUrls": [{"url": depop_search_url}],
        "category": "hoodies",
        "categories": ["hoodies", "crewnecks", "zip-ups", "sweatshirts"],
        "maxItems": config.max_items,
        "maxResults": config.max_items,
        "limit": config.max_items,
    }

    if config.actor_input_path:
        actor_input_file = Path(config.actor_input_path)
        custom_input = json.loads(actor_input_file.read_text(encoding="utf-8"))
        base_input.update(custom_input)
    return base_input


def run_actor(config: ApifyConfig) -> dict[str, Any]:
    endpoint = f"{config.api_base_url}/acts/{config.actor_id}/runs"
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {config.api_token}"},
        json=_build_actor_input(config),
        timeout=120,
    )
    _raise_for_status_with_context(response, actor_id=config.actor_id)
    return response.json()["data"]


def wait_for_run(config: ApifyConfig, run_id: str) -> dict[str, Any]:
    endpoint = f"{config.api_base_url}/actor-runs/{run_id}"
    for _ in range(config.max_attempts):
        response = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {config.api_token}"},
            timeout=60,
        )
        _raise_for_status_with_context(response, actor_id=config.actor_id, run_id=run_id)
        run = response.json()["data"]
        if run.get("status") in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}:
            return run
        import time

        time.sleep(config.poll_seconds)
    raise TimeoutError(
        f"Apify actor run {run_id} did not finish within the expected window "
        f"({config.max_attempts * config.poll_seconds} seconds)."
    )


def fetch_dataset_items(config: ApifyConfig, dataset_id: str) -> list[dict[str, Any]]:
    endpoint = f"{config.api_base_url}/datasets/{dataset_id}/items"
    response = requests.get(
        endpoint,
        params={"format": "json", "clean": "true"},
        headers={"Authorization": f"Bearer {config.api_token}"},
        timeout=120,
    )
    _raise_for_status_with_context(response, actor_id=config.actor_id, dataset_id=dataset_id)
    payload = response.json()
    return payload if isinstance(payload, list) else payload.get("items", [])


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "No response body returned."

    if isinstance(payload, dict):
        if isinstance(payload.get("error"), dict):
            message = payload["error"].get("message")
            if message:
                return message
        message = payload.get("message")
        if message:
            return str(message)
    return json.dumps(payload)


def _raise_for_status_with_context(
    response: requests.Response,
    actor_id: str,
    run_id: str | None = None,
    dataset_id: str | None = None,
) -> None:
    if response.ok:
        return

    message = _extract_error_message(response)
    context_parts = [f"actor={actor_id}"]
    if run_id:
        context_parts.append(f"run={run_id}")
    if dataset_id:
        context_parts.append(f"dataset={dataset_id}")

    if response.status_code == 403:
        hint = (
            "Apify rejected access to this actor. Check that APIFY_API_TOKEN is valid, "
            "the actor ID exists, and your account/token has permission to run it."
        )
    elif response.status_code == 404:
        hint = "Apify could not find that actor or dataset. Double-check APIFY_ACTOR_ID."
    else:
        hint = "Apify request failed."

    raise ApifyRequestError(
        f"{hint} ({', '.join(context_parts)}, status={response.status_code}) Details: {message}"
    )


def _first_non_empty(item: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _detect_hoodie_type(title: str, description: str, category: str) -> str | None:
    combined = " ".join([title, description, category]).lower()
    if "zip" in combined:
        return "zip-up"
    if "crew" in combined:
        return "crewneck"
    if "oversized" in combined:
        return "oversized sweatshirt"
    if HOODIE_PATTERN.search(combined):
        return "hoodie"
    return None


def _normalize_numeric(value: Any) -> Any:
    if isinstance(value, dict):
        value = _first_non_empty(value, ["amount", "value", "price"])
    if isinstance(value, (int, float)) or value is None:
        return value
    if isinstance(value, str):
        stripped = value.replace(",", "")
        match = re.search(r"-?\d+(\.\d+)?", stripped)
        if match:
            number = float(match.group(0))
            return int(number) if number.is_integer() else number
    return value


def _parse_images(value: Any) -> list[str]:
    if isinstance(value, list):
        output = []
        for item in value:
            if isinstance(item, str):
                output.append(item)
            elif isinstance(item, dict):
                maybe_url = _first_non_empty(item, ["url", "src", "imageUrl", "original"])
                if maybe_url:
                    output.append(str(maybe_url))
        return output
    if isinstance(value, str) and value:
        return [value]
    return []


def normalize_apify_items(items: list[dict[str, Any]], actor_id: str, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in items:
        title = str(_first_non_empty(item, ["title", "name", "listing_title"], "") or "")
        description = str(_first_non_empty(item, ["description", "listing_description", "text"], "") or "")
        category = str(_first_non_empty(item, ["category", "primaryCategory", "subcategory"], "") or "")
        brand = _first_non_empty(item, ["brand", "designer", "label"])
        size = _first_non_empty(item, ["size", "sizes"])
        condition = _first_non_empty(item, ["condition", "conditionText"])
        color = _first_non_empty(item, ["color", "colour"])
        price = _normalize_numeric(_first_non_empty(item, ["price", "priceAmount", "currentPrice"]))
        shipping_price = _normalize_numeric(_first_non_empty(item, ["shipping_price", "shippingPrice", "shipping"]))
        likes = _normalize_numeric(_first_non_empty(item, ["likes", "likesCount", "favoriteCount"]))
        comments_count = _normalize_numeric(_first_non_empty(item, ["comments_count", "commentsCount"]))
        seller_rating = _normalize_numeric(_first_non_empty(item, ["seller_rating", "sellerRating", "rating"]))
        seller_followers = _normalize_numeric(_first_non_empty(item, ["seller_followers", "sellerFollowers", "followers"]))
        discounted = _first_non_empty(item, ["discounted", "isDiscounted", "onSale"], False)
        listed_date = _first_non_empty(item, ["listed_date", "date_listed", "listedAt", "createdAt"])
        sold_date = _first_non_empty(item, ["sold_date", "date_sold", "soldAt"])
        sold_status = _first_non_empty(item, ["sold_status", "isSold", "sold"], None)
        image_urls = _parse_images(_first_non_empty(item, ["image_urls", "imageUrls", "images", "photos"], []))
        source_url = _first_non_empty(item, ["source_url", "url", "itemUrl", "listingUrl"])

        hoodie_type = _detect_hoodie_type(title, description, category or "")
        combined_text = " ".join([title, description, category or ""])
        if not HOODIE_PATTERN.search(combined_text):
            continue

        rows.append(
            {
                "listing_id": _first_non_empty(item, ["listing_id", "id", "listingId", "itemId"]),
                "source_platform": "depop",
                "source_actor_id": actor_id,
                "source_run_id": run_id,
                "source_url": source_url,
                "title": title,
                "description": description,
                "price": price,
                "shipping_price": shipping_price,
                "total_buyer_cost": None,
                "brand": brand,
                "category": category or "hoodies",
                "hoodie_type": hoodie_type,
                "size": size,
                "condition": condition,
                "color": color,
                "likes": likes,
                "comments_count": comments_count,
                "seller_rating": seller_rating,
                "seller_followers": seller_followers,
                "discounted": discounted,
                "listed_date": listed_date,
                "sold_date": sold_date,
                "sold_status": sold_status,
                "image_urls": json.dumps(image_urls),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=CANONICAL_COLUMNS)
    frame = normalize_columns(frame)
    ordered_columns = [column for column in CANONICAL_COLUMNS if column in frame.columns]
    extra_columns = [column for column in frame.columns if column not in ordered_columns]
    frame = frame[ordered_columns + extra_columns]
    return frame


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_raw_and_normalized(
    raw_items: list[dict[str, Any]],
    normalized_frame: pd.DataFrame,
    output_dir: str | Path = DEFAULT_RAW_DIR,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp_slug()
    raw_path = output_path / f"apify_depop_raw_{stamp}.json"
    csv_path = output_path / f"apify_depop_hoodies_{stamp}.csv"

    raw_path.write_text(json.dumps(raw_items, indent=2), encoding="utf-8")
    normalized_frame.to_csv(csv_path, index=False)
    return raw_path, csv_path


def fetch_and_store_apify_data(env_path: str | Path | None = None, output_dir: str | Path = DEFAULT_RAW_DIR) -> tuple[Path, Path]:
    config = load_apify_config(env_path=env_path)
    initial_run = run_actor(config)
    print(f"Started Apify run {initial_run['id']} for actor {config.actor_id}")
    final_run = wait_for_run(config, initial_run["id"])

    if final_run.get("status") != "SUCCEEDED":
        status = final_run.get("status")
        status_message = final_run.get("statusMessage") or "No status message returned."
        run_id = final_run.get("id", initial_run["id"])
        console_url = f"https://console.apify.com/actors/runs/{run_id}"
        raise RuntimeError(
            f"Apify actor run ended with status {status}. "
            f"Run ID: {run_id}. "
            f"Status message: {status_message}. "
            f"Inspect the run log in Apify: {console_url}"
        )

    dataset_id = final_run.get("defaultDatasetId")
    if not dataset_id:
        raise RuntimeError("Apify run finished without a default dataset ID.")

    raw_items = fetch_dataset_items(config, dataset_id)
    normalized_frame = normalize_apify_items(raw_items, actor_id=config.actor_id, run_id=final_run["id"])
    return save_raw_and_normalized(raw_items, normalized_frame, output_dir=output_dir)
