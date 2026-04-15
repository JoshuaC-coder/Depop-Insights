# Depop Insights

Depop Insights upgrades the original terminal-based predictor into a multi-category pricing and listing-intelligence foundation with:

- a Manifest V3 Chrome extension MVP
- a FastAPI backend
- a reusable ML pipeline for training and inference
- clearer product structure and developer documentation

Current training scope:

- Primary supervised dataset: Mercari Price Suggestion Challenge data on Kaggle (`train.tsv`)
- Primary trained target: expected sale price / suggested price range
- Sell-through probability and days-to-sell remain heuristic until a reliable labeled resale dataset with sold outcomes is added
- Training now defaults to all supported categories; use `--hoodies-only` if you want the older niche slice

## Repository layout

```text
Depop-Insights/
  backend/
  data/
  docs/
  extension/
  legacy_v1/
  ml/
  README.md
```

## What is live now

- `backend/` exposes `/health` and `/predict`
- `extension/` can scrape a Depop listing page and send the result to the backend
- `ml/` includes preprocessing, feature engineering, training, evaluation, and artifact persistence code
- `legacy_v1/` preserves the original CLI predictor, dataset, and model artifacts

## Running locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r ml/requirements.txt
uvicorn backend.app.main:app --reload
```

Then load `extension/` as an unpacked Chrome extension and open a Depop listing page.

## Training models

You can either place a real listing CSV in `data/raw/` manually or fetch one from Apify.

### Apify fetch

Create `.env` from `.env.example`, set `APIFY_API_TOKEN` and `APIFY_ACTOR_ID`, then run:

```bash
python3 fetch_apify_data.py
```

This saves both the raw Apify response and a normalized training-ready CSV into `data/raw/`.

### Train from CSV/TSV

Run:

```bash
python3 -m ml.scripts.train_classifier data/raw/your_hoodie_dataset.csv
python3 -m ml.scripts.run_evaluation data/raw/your_hoodie_dataset.csv
```

Artifacts will be written to `data/models/`, and evaluation reports to `data/processed/`.

Mercari Kaggle example:

```bash
python3 -m ml.scripts.train_classifier /path/to/train.tsv
python3 -m ml.scripts.run_evaluation /path/to/train.tsv
```

Optional niche slice:

```bash
python3 -m ml.scripts.train_classifier --hoodies-only /path/to/train.tsv
```

## Important honesty note

This repository does not fabricate final production results. If only price-labeled artifacts exist in `data/models/`, the backend uses those for price guidance and keeps sell-speed outputs heuristic.

## Key docs

- [Architecture](./docs/architecture.md)
- [Apify integration](./docs/apify-integration.md)
- [Data schema](./docs/data-schema.md)
- [Local setup](./docs/local-setup.md)
- [Extension workflow](./docs/extension-workflow.md)
- [Model training](./docs/model-training.md)
- [Sample training CSV schema](./docs/sample-training-csv-schema.md)
