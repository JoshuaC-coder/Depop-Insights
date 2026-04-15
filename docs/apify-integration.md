# Apify Integration

## Environment setup

Create a `.env` file in the repo root and add:

```env
APIFY_API_TOKEN=your_apify_api_token_here
APIFY_ACTOR_ID=your-username~your-depop-actor
APIFY_SEARCH_QUERY=hoodie sweatshirt crewneck zip up oversized sweatshirt
APIFY_MAX_ITEMS=200
```

Optional:

```env
APIFY_ACTOR_INPUT_PATH=./path/to/actor_input.json
```

`APIFY_ACTOR_ID` is intentionally configurable so you can switch between different Depop actors without changing code.

## Fetch command

```bash
python3 fetch_apify_data.py
```

The script will:

1. Load your Apify credentials from `.env`
2. Trigger the configured actor run
3. Wait for the run to finish
4. Download the actor dataset items
5. Save the raw payload into `data/raw/`
6. Normalize hoodie and sweatshirt listings into a training-ready CSV in `data/raw/`

## Output files

Example output:

- `data/raw/apify_depop_raw_20260414T230000Z.json`
- `data/raw/apify_depop_hoodies_20260414T230000Z.csv`

The CSV uses the current canonical schema so it can be passed directly into:

```bash
python3 -m ml.scripts.train_classifier data/raw/apify_depop_hoodies_20260414T230000Z.csv
```

## Actor input notes

Because different Apify actors use different input field names, the fetcher sends a broad listing-focused input payload with keys such as:

- `query`
- `queries`
- `search`
- `searchTerms`
- `searchStringsArray`
- `startUrls`
- `category`
- `categories`
- `maxItems`

If your chosen actor expects different keys, create a JSON file and point `APIFY_ACTOR_INPUT_PATH` at it. Those custom values override the defaults.

For the `lexis-solutions~depop-scraper` actor specifically, `startUrls` is required. The fetcher now provides a Depop search URL automatically based on `APIFY_SEARCH_QUERY`.

## Important limitations

- The fetcher keeps only hoodie, crewneck, zip-up, and sweatshirt-like rows in the normalized CSV.
- Raw JSON is always preserved so you can revisit the source payload if an actor changes shape.
- Some fields such as seller rating or followers will remain blank if the actor does not return them.
