# Architecture Overview

Depop Insights is split into three primary layers:

- `extension/`: a Manifest V3 Chrome extension that runs on Depop listing pages, scrapes visible listing data, and renders pricing and sell-speed outputs for the user.
- `backend/`: a FastAPI service that validates incoming listing payloads, loads trained ML artifacts when available, and falls back to transparent heuristic predictions when label coverage is incomplete.
- `ml/`: reusable data ingestion, preprocessing, feature engineering, training, evaluation, and inference code for multi-category resale pricing and listing-quality modeling.

Current MVP behavior:

- The extension is targeted to Depop listing pages.
- The backend exposes `/health` and `/predict`.
- Mercari price-labeled data is the current supervised training anchor.
- If only price artifacts are present in `data/models/`, the backend serves trained price guidance plus heuristic sell-speed outputs.
- If no trained model artifacts are present in `data/models/`, the backend serves heuristic baseline outputs so the full product can still be tested end-to-end.

Legacy isolation:

- The former V1 CLI scripts and sample CSV are stored under `legacy_v1/`.
- The active app does not depend on legacy artifacts for runtime behavior.
