# Data Layout

- `raw/`: place future scraper exports or curated CSV files here.
- `processed/`: generated reports, cleaned datasets, and evaluation outputs.
- `models/`: trained classifier/regressor artifacts consumed by the backend.

The original V1 dataset remains isolated under `legacy_v1/` and is intentionally not mixed into the active V2 data flow.
