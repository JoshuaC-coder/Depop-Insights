# Sample Training Schema

The normalized Apify output is designed to match this CSV header:

```csv
listing_id,source_platform,source_actor_id,source_run_id,source_url,title,description,price,shipping_price,total_buyer_cost,brand,category,hoodie_type,size,condition,color,likes,comments_count,seller_rating,seller_followers,discounted,listed_date,sold_date,sold_status,image_urls
```

Example row:

```csv
123456,depop,username~depop-actor,abc123,https://www.depop.com/products/example,Carhartt heavyweight hoodie,Oversized faded hoodie with minor wear,42.0,7.5,49.5,Carhartt,hoodies,hoodie,L,good,brown,14,2,4.9,1800,false,2026-04-02T00:00:00Z,2026-04-18T00:00:00Z,true,"[""https://example.com/image1.jpg"",""https://example.com/image2.jpg""]"
```

Field notes:

- `listing_id`: source listing identifier when available
- `source_actor_id`: actor used for scraping
- `source_run_id`: Apify run ID for traceability
- `listed_date` and `sold_date`: ISO-like timestamps preferred
- `image_urls`: JSON-encoded string of image URLs
- `sold_status`: `true` or `false`

This schema is compatible with the current preprocessing pipeline.

Mercari Kaggle input is also supported directly as TSV with columns like:

```tsv
train_id	name	item_condition_id	category_name	brand_name	price	shipping	item_description
```

Relevant mappings:

- `name -> title`
- `item_description -> description`
- `brand_name -> brand`
- `category_name -> category`
- `item_condition_id -> condition`
- `shipping -> shipping_price`
- `train_id -> listing_id`
