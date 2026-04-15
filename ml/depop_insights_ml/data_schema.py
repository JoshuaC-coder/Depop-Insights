from __future__ import annotations

from dataclasses import dataclass


CANONICAL_COLUMNS = [
    "listing_id",
    "source_platform",
    "source_actor_id",
    "source_run_id",
    "source_url",
    "title",
    "description",
    "price",
    "shipping_price",
    "total_buyer_cost",
    "brand",
    "category",
    "hoodie_type",
    "size",
    "condition",
    "color",
    "likes",
    "comments_count",
    "seller_rating",
    "seller_followers",
    "discounted",
    "listed_date",
    "sold_date",
    "sold_status",
    "image_urls",
]

COLUMN_ALIASES = {
    "id": "listing_id",
    "listingId": "listing_id",
    "itemId": "listing_id",
    "train_id": "listing_id",
    "test_id": "listing_id",
    "url": "source_url",
    "itemUrl": "source_url",
    "date_listed": "listed_date",
    "date_sold": "sold_date",
    "name": "title",
    "listing_title": "title",
    "item_description": "description",
    "listing_description": "description",
    "brand_name": "brand",
    "category_name": "category",
    "ship_price": "shipping_price",
    "shipping": "shipping_price",
    "shippingPrice": "shipping_price",
    "totalBuyerCost": "total_buyer_cost",
    "item_condition_id": "condition",
    "sellerFollowers": "seller_followers",
    "followers": "seller_followers",
    "sellerRating": "seller_rating",
    "rating": "seller_rating",
    "commentsCount": "comments_count",
    "comments": "comments_count",
    "imageUrls": "image_urls",
    "isSold": "sold_status",
}

TEXT_COLUMNS = ["title", "description"]
CATEGORICAL_COLUMNS = ["brand", "category", "hoodie_type", "size", "condition", "color"]
NUMERIC_COLUMNS = [
    "price",
    "shipping_price",
    "total_buyer_cost",
    "likes",
    "comments_count",
    "seller_rating",
    "seller_followers",
]


@dataclass(frozen=True)
class TargetColumns:
    sold_within_30_days: str = "sold_within_30_days"
    days_to_sell: str = "days_to_sell"
    expected_sale_price: str = "expected_sale_price"
