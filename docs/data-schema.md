# Data Schema Spec

Canonical fields expected by the active pipeline:

- `title`
- `description`
- `price`
- `shipping_price`
- `total_buyer_cost`
- `brand`
- `category`
- `hoodie_type`
- `size`
- `condition`
- `color`
- `likes`
- `comments_count`
- `seller_rating`
- `seller_followers`
- `discounted`
- `listed_date`
- `sold_date`
- `sold_status`
- `image_urls`

Derived targets:

- `sold_within_30_days`: optional binary classification label derived from `listed_date` and `sold_date`
- `days_to_sell`: optional regression target
- `expected_sale_price`: primary regression target, currently using observed `price` as the baseline label

Notes:

- The preprocessing layer supports a few aliases such as `date_listed -> listed_date` and `date_sold -> sold_date`.
- Mercari-style fields are also supported, including `name -> title`, `brand_name -> brand`, `category_name -> category`, `item_description -> description`, and `item_condition_id -> condition`.
- Training now defaults to all supported categories. A hoodie-only slice is still available via the CLI flag.
- Relative market features are computed only from available data and improve automatically when the dataset grows.
