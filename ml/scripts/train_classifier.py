from __future__ import annotations

import argparse

from ml.depop_insights_ml.training import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Depop Insights price and optional sell-through models.")
    parser.add_argument("csv_path", help="Path to the raw listing CSV or Mercari TSV.")
    parser.add_argument("--hoodies-only", action="store_true", help="Restrict training to hoodie/sweatshirt-like items.")
    parser.add_argument("--no-catboost", action="store_true", help="Use sklearn fallbacks even if CatBoost is installed.")
    args = parser.parse_args()

    artifacts = train_all(
        csv_path=args.csv_path,
        hoodie_only=args.hoodies_only,
        use_catboost=not args.no_catboost,
    )
    print(f"Trained on {artifacts.dataset_rows} rows.")
    print(artifacts.metrics)


if __name__ == "__main__":
    main()
