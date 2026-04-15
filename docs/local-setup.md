# Local Setup

## Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r ml/requirements.txt
uvicorn backend.app.main:app --reload
```

## Extension

1. Open `chrome://extensions`
2. Enable Developer Mode
3. Click `Load unpacked`
4. Select the `extension/` folder
5. Open a Depop listing page and run the popup

## Data and models

- Put future listing CSV exports in `data/raw/`
- Or configure Apify in `.env` and run `python3 fetch_apify_data.py`
- Run the ML training scripts to populate `data/models/`
- If `data/models/` is empty, the backend will return heuristic fallback predictions
