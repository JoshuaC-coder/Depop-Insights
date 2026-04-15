# Model Training Workflow

## Training inputs

Use a real listing CSV export from Apify or another scraper, or bootstrap with the Mercari Price Suggestion Challenge data. Mercari is the current preferred supervised starting point because it contains price labels at scale, even though it does not include sell-through timing labels.

Recommended fields:

- title and description text
- price and shipping price
- brand, category, hoodie type, size, condition, color
- likes and seller metrics
- listed and sold timestamps

## Commands

Fetch data from Apify:

```bash
python3 fetch_apify_data.py
```

Train on a normalized CSV/TSV:

```bash
python3 -m ml.scripts.train_classifier data/raw/your_hoodie_dataset.csv
python3 -m ml.scripts.run_evaluation data/raw/your_hoodie_dataset.csv
```

Mercari Kaggle example:

```bash
python3 -m ml.scripts.train_classifier /path/to/train.tsv
python3 -m ml.scripts.run_evaluation /path/to/train.tsv
```

Optional niche slice:

```bash
python3 -m ml.scripts.train_classifier --hoodies-only /path/to/train.tsv
```

## Modeling notes

- The pipeline builds structured features, keyword features, date features, and TF-IDF text features.
- The loader supports both `.csv` and `.tsv` inputs.
- CatBoost is preferred when installed.
- Sklearn fallback models are used if CatBoost is unavailable.
- Evaluation reports include price regression metrics by default and classification / days-to-sell metrics where enough labels exist.

## Honest limitations

- No real production performance numbers should be claimed until those commands are run on real category-relevant data.
- Relative market features become more trustworthy as the dataset grows.
- If a dataset contains active listings only and does not include sold timestamps or sold-status outcomes, the sell-through tasks are skipped by design and the project behaves as a price-first model stack.
