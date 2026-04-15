from __future__ import annotations

import argparse

from backend.app.config import MODELS_DIR, PROCESSED_DIR
from ml.depop_insights_ml.evaluate import write_metrics_report
from ml.depop_insights_ml.training import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models and write a metrics report.")
    parser.add_argument("csv_path", help="Path to the raw listing CSV or Mercari TSV.")
    parser.add_argument("--hoodies-only", action="store_true", help="Restrict evaluation/training to hoodie-like items.")
    args = parser.parse_args()

    artifacts = train_all(csv_path=args.csv_path, models_dir=MODELS_DIR, hoodie_only=args.hoodies_only)
    write_metrics_report(artifacts.metrics, PROCESSED_DIR / "latest_metrics.json")
    print("Metrics written to data/processed/latest_metrics.json")


if __name__ == "__main__":
    main()
